# Import built-in module
import argparse
import warnings

import torch.nn

warnings.filterwarnings(action='ignore')

# Import torch
import torch.optim as optim
import torch.distributed as dist

# Import Custom Utils
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# attack loader
from attack.fastattack import attack_loader

# Accelerating forward and backward
from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--NAME', default='DAML-TRADES', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int) # 12 for vit
parser.add_argument('--gpu', default='4,5,6,7', type=str)
parser.add_argument('--port', default="12356", type=str)


# transformer parameter
parser.add_argument('--patch_size', default=16, type=int, help='4/16/32')
parser.add_argument('--img_resize', default=224, type=int, help='32/224')
parser.add_argument('--tran_type', default='base', type=str, help='tiny/small/base/large/huge')
parser.add_argument('--warmup-steps', default=500, type=int)
parser.add_argument("--num_steps", default=10000, type=int)

# learning parameter
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float) #3e-2 for ViT
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--test_batch_size', default=64, type=float)
parser.add_argument('--pretrain', default=False, type=bool)

# attack parameter
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=8/255, type=float)
parser.add_argument('--steps', default=10, type=int)
args = parser.parse_args()

# the number of gpus for multi-process
gpu_list = list(map(int, args.gpu.split(',')))
ngpus_per_node = len(gpu_list)

# cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = args.port

# global best_acc
best_acc = 0

# Mix Training
scaler = GradScaler()

# Tensorboard settings
counter = 0
log_dir = './logs'
check_dir(log_dir)

# make checkpoint folder and set checkpoint name for saving
if not os.path.isdir(f'checkpoint'): os.mkdir(f'checkpoint')
if not os.path.isdir(f'checkpoint/daml_trades'): os.mkdir(f'checkpoint/daml_trades')
if not os.path.isdir(f'checkpoint/daml_trades/{args.dataset}'): os.mkdir(f'checkpoint/daml_trades/{args.dataset}')
if args.network in transformer_list:
    saving_ckpt_name = f'./checkpoint/daml_trades/{args.dataset}/{args.dataset}_daml_trades_{args.network}_{args.tran_type}_patch{args.patch_size}_{args.img_resize}_best.t7'
else:
    saving_ckpt_name = f'./checkpoint/daml_trades/{args.dataset}/{args.dataset}_daml_trades_{args.network}{args.depth}_best.t7'

def train(net, trainloader, optimizer, lr_scheduler, scaler, attack, rank):
    global counter
    net.train()

    train_loss = 0
    correct, adv_correct = 0, 0
    total, total_g = 0, 0

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), leave=True)
    for batch_idx, (inputs, targets) in prog_bar:

        # clean sample splitting
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inputs = attack(inputs, targets)

        # inputs
        inputs1, inputs2 = inputs.split(args.batch_size // 2)
        adv_inputs1, adv_inputs2 = adv_inputs.split(args.batch_size // 2)
        targets1, targets2 = targets.split(args.batch_size // 2)

        # f optimization
        optimizer.zero_grad()
        with autocast():
            # network propagation
            adv_outputs1 = net(adv_inputs1)
            outputs1 = net(inputs1)

            # base loss
            base_loss = trades_loss(outputs1, adv_outputs1, targets1)

            # network propagation
            adv_outputs2 = net(adv_inputs2)

            # attack
            is_attack2 = adv_outputs2.max(1)[1] != targets2
            is_not_attack2 = ~is_attack2

            # Theta: Target
            Y_do_T = (targets2.shape[0] / is_attack2.sum()-1) * F.cross_entropy(adv_outputs2[is_attack2], targets2[is_attack2])
            Y_do_g = (targets2.shape[0] / is_not_attack2.sum()-1) * F.cross_entropy(adv_outputs2[is_not_attack2], targets2[is_not_attack2])
            dml_loss1 = Y_do_T - Y_do_g

            # Theta: Non-Target
            Y_do_T = (targets2.shape[0] / is_attack2.sum()-1) * non_target_dml(adv_outputs2[is_attack2], targets2[is_attack2])
            Y_do_g = (targets2.shape[0] / is_not_attack2.sum()-1) * non_target_dml(adv_outputs2[is_not_attack2], targets2[is_not_attack2])
            dml_loss2 = Y_do_T - Y_do_g

            # DML loss
            dml_loss = dml_loss1 + dml_loss2

            # Total Loss
            loss = base_loss + dml_loss.abs()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # scheduling for Cyclic LR
        lr_scheduler.step()

        train_loss += loss.item()

        # for test
        with autocast():

            adv_outputs = net(adv_inputs)
            outputs = net(inputs)

        _, adv_predicted = adv_outputs.max(1)
        _, predicted = outputs.max(1)

        total += targets.size(0)
        total_g += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        adv_correct += adv_predicted.eq(targets).sum().item()

        desc = ('[Tr/lr=%.3f] Loss: (F/G) %.3f | Acc: (Clean) %.2f%% | Acc: (PGD) %.2f%%' %
                (lr_scheduler.get_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, 100. * adv_correct / total))
        prog_bar.set_description(desc, refresh=True)

def trades_loss(logits,
                logits_adv,
                targets):
    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    loss_natural = F.cross_entropy(logits_adv, targets)
    loss_robust = (1.0 / logits.shape[0]) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
    loss = loss_natural + float(2) * loss_robust
    return loss

def test(net, testloader, attack, rank):
    global best_acc
    net.eval()
    test_loss = 0
    correct= 0
    total = 0

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()

        # Accerlating forward propagation
        with autocast():
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = (f'[Test/Clean] Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}')
        prog_bar.set_description(desc, refresh=True)

    # Save clean acc.
    clean_acc = 100. * correct / total

    test_loss = 0
    correct = 0
    total = 0


    prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inputs = attack(inputs, targets)

        # Accerlating forward propagation
        with autocast():
            adv_outputs = net(adv_inputs)
            outputs = net(inputs)
            loss = F.cross_entropy(adv_outputs, targets)
        _, adv_predicted = adv_outputs.max(1)
        _, predicted = outputs.max(1)

        test_loss += loss.item()
        total += targets.size(0)
        correct += adv_predicted.eq(targets).sum().item()

        desc = ('[Test/PGD] Loss: %.3f | Acc: (F) %.3f%%'
                % (test_loss / (batch_idx + 1), 100. * correct / total))
        prog_bar.set_description(desc, refresh=True)

    # Save adv acc.
    adv_acc = 100. * correct / total

    # compute acc
    acc = (clean_acc + adv_acc) / 2

    # current accuracy print
    rprint(f'Current Accuracy is {clean_acc:.2f}/{adv_acc:.2f}!!', rank)

    # saving checkpoint
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
        }

        torch.save(state, saving_ckpt_name)
        rprint(f'Saving~ {saving_ckpt_name}', rank)

        # update best acc
        best_acc = acc

def main_worker(rank, ngpus_per_node=ngpus_per_node):
    # print configuration
    print_configuration(args, rank)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # DDP environment settings
    print(f'Use GPU: {gpu_list[rank]} for training')
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)

    # Load ADV or Standard Network
    if args.network in transformer_list:
        pretrain_ckpt_name = f'checkpoint/adv/{args.dataset}/{args.dataset}_adv_{args.network}_{args.tran_type}_patch{args.patch_size}_{args.img_resize}_best.t7'
        checkpoint = torch.load(pretrain_ckpt_name, map_location=torch.device(torch.cuda.current_device()))
    else:
        # adv
        pretrain_ckpt_name = f'checkpoint/adv/{args.dataset}/{args.dataset}_adv_{args.network}{args.depth}_best.t7'
        checkpoint = torch.load(pretrain_ckpt_name, map_location=torch.device(torch.cuda.current_device()))

    # network f
    # init model and Distributed Data Parallel
    net = get_network(network=args.network, depth=args.depth, dataset=args.dataset, tran_type=args.tran_type,
                      img_size=args.img_resize, patch_size=args.patch_size, pretrain=args.pretrain)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.to(memory_format=torch.channels_last).cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=[rank])
    net.load_state_dict(checkpoint['net'])

    rprint(f'==> {pretrain_ckpt_name}', rank)
    rprint('==> Successfully Loaded Standard checkpoint..', rank)

    # upsampling for transformer
    upsample = True if args.network in transformer_list else False

    # fast dataloader
    trainloader, testloader, decoder = get_fast_dataloader(dataset=args.dataset, train_batch_size=args.batch_size,
                                                           test_batch_size=args.test_batch_size, upsample=upsample)

    # Attack loader
    if args.dataset == 'imagenet':
        rprint('Fast FGSM training', rank)
        attack = attack_loader(net=net, attack='fgsm_train', eps=args.eps/4, steps=args.steps)
    elif args.dataset == 'tiny':
        if args.network in transformer_list:
            rprint('PGD and FGSM MIX training', rank)
            pgd_attack = attack_loader(net=net, attack='pgd', eps=args.eps / 2, steps=args.steps)
            fgsm_attack = attack_loader(net=net, attack='fgsm_train', eps=args.eps / 2, steps=args.steps)
            attack = MixAttack(net=net, slowattack=pgd_attack, fastattack=fgsm_attack, train_iters=len(trainloader))
        else:
            rprint('Fast FGSM training', rank)
            attack = attack_loader(net=net, attack='fgsm_train', eps=args.eps / 2, steps=args.steps)
    else:
        rprint('PGD training', rank)
        attack = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps)


    # init optimizer and lr scheduler
    # optimizer network concurrent
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=args.learning_rate,
                                                    step_size_up=int(round(args.epochs/5))*len(trainloader),
                                                    step_size_down=args.epochs*len(trainloader)-int(round(args.epochs/5))*len(trainloader))

    for epoch in range(args.epochs):
        rprint(f'\nEpoch: {epoch+1}', rank)
        if args.dataset == "imagenet":
            if args.network in transformer_list:
                res = args.img_resize
            else:
                res = get_resolution(epoch=epoch, min_res=160, max_res=192,
                                     start_ramp=int(math.floor(args.epochs * 0.5)),
                                     end_ramp=int(math.floor(args.epochs * 0.7)))
            decoder.output_size = (res, res)

        train(net, trainloader, optimizer, lr_scheduler, scaler, attack, rank)
        test(net, testloader, attack, rank)

def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()