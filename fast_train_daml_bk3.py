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
from utils.scheduler import WarmupCosineSchedule
from models.generator import weights_init

# attack loader
# from attack.attack import attack_loader
from attack.fastattack import attack_loader

# Accelerating forward and backward
from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--NAME', default='DAML', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int) # 12 for vit
parser.add_argument('--gpu', default='4,5,6,7', type=str)
parser.add_argument('--port', default="12357", type=str)

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
scaler_logistic = GradScaler()
scaler_total = GradScaler()

# make checkpoint folder and set checkpoint name for saving
if not os.path.isdir(f'checkpoint'): os.mkdir(f'checkpoint')
if not os.path.isdir(f'checkpoint/daml'): os.mkdir(f'checkpoint/daml')
if not os.path.isdir(f'checkpoint/daml/{args.dataset}'): os.mkdir(f'checkpoint/daml/{args.dataset}')
if args.network in transformer_list:
    saving_ckpt_name = f'./checkpoint/daml/{args.dataset}/{args.dataset}_daml_{args.network}_{args.tran_type}_patch{args.patch_size}_{args.img_resize}_best.t7'
else:
    saving_ckpt_name = f'./checkpoint/daml/{args.dataset}/{args.dataset}_daml_{args.network}{args.depth}_best.t7'


def train(net, net_logistic, trainloader, optimizer_list, lr_scheduler_list, scaler_list, attack, rank):
    optimizer, optimizer_logistic,  optimizer_total = optimizer_list
    lr_scheduler, lr_scheduler_logistic, lr_scheduler_total = lr_scheduler_list
    scaler, scaler_logistic, scaler_total = scaler_list

    net.train()
    net_logistic.train()

    train_loss, train_loss_logistic, train_loss_theta= 0, 0, 0
    correct, correct_logistic, correct_sim1, correct_sim2 = 0, 0, 0, 0
    total, total_sim1, total_sim2 = 0, 0, 0


    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), leave=True)
    for batch_idx, (inputs, targets) in prog_bar:

        # input setting
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inputs = attack(inputs, targets)

        # sample splitting
        inputs1, inputs2 = inputs.split(args.batch_size//2)
        adv_inputs1, adv_inputs2 = adv_inputs.split(args.batch_size // 2)
        targets1, targets2 = targets.split(args.batch_size//2)


        # DAML STEP [1]
        # (1-1): optimizer_logistic init
        # optimizer_logistic.zero_grad()
        # with autocast():
        #     adv_outputs1 = net(adv_inputs1)
        #     outputs1 = net(inputs1)
        #     outputs_logistic1 = net_logistic(inputs1)
        #
        #     adv_predicted1 = adv_outputs1.max(1)[1]
        #     predicted1 = outputs1.max(1)[1]
        #
        #     # logistic_target1 = (adv_predicted1 != predicted1).float().view(-1,1)
        #     logistic_target1 = (adv_predicted1 != targets1).float().view(-1,1)
        #     loss_logistic = torch.nn.BCEWithLogitsLoss()(outputs_logistic1, logistic_target1)
        #
        # scaler_logistic.scale(loss_logistic).backward()
        # scaler_logistic.step(optimizer_logistic)
        # scaler_logistic.update()

        # (1-2): optimizer init
        optimizer.zero_grad()
        with autocast():
            adv_outputs1 = net(adv_inputs1)
            loss = F.cross_entropy(adv_outputs1, targets1)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # # DAML STEP [2]
        # # (2): optimizerD init
        optimizer_total.zero_grad()
        with autocast():
            adv_outputs2 = net(adv_inputs2)
            outputs2 = net(inputs2)

            l_adv = F.cross_entropy(adv_outputs2, targets2, reduction='none')
            l_nat = F.cross_entropy(outputs2, targets2, reduction='none')

            theta1 = (adv_outputs2.softmax(dim=1) * ((adv_outputs2.softmax(dim=1)+1e-3).log() - (outputs2.softmax(dim=1)+1e-3).log())).sum(dim=1)

            # first theta2
            theta2 = 1/(adv_outputs2.softmax(dim=1)[:, targets2]+1e-3).detach() * F.cross_entropy(adv_outputs2, targets2, reduction='none') \
                     - 1/(1-adv_outputs2.softmax(dim=1)[:, targets2]+1e-3).detach() * F.cross_entropy(outputs2, targets2, reduction='none')

            # second theta2
            theta2 = l_adv - l_nat

            theta = theta1 + theta2.abs()
            loss_theta = theta.mean()

        scaler_total.scale(loss_theta).backward()
        scaler_total.step(optimizer)
        scaler_total.update()

        # scheduling for Cyclic LR
        lr_scheduler.step()
        lr_scheduler_total.step()

        train_loss += loss.item()
        train_loss_theta += loss_theta.item()

        # for test
        with autocast():
            adv_outputs1 = net(adv_inputs1)
            outputs1 = net(inputs1)
        _, adv_predicted1 = adv_outputs1.max(1)
        _, predicted1 = outputs1.max(1)

        total += targets1.size(0)
        correct += predicted1.eq(targets1).sum().item()

        desc = ('[Tr/lrFG=%.3f/%.3f] Loss: (F) %.3f | (G) %.3f | (theta) %.3f | Acc: (F) %.2f%% | l_adv/l_nat: %.2f%%' %
                (lr_scheduler.get_lr()[0], lr_scheduler_logistic.get_lr()[0], train_loss / (batch_idx + 1), train_loss_logistic / (batch_idx + 1),
                 train_loss_theta / (batch_idx + 1), 100. * correct / total, l_adv.mean().item()/l_nat.mean().item()))
        prog_bar.set_description(desc, refresh=True)

def test(net, net_logistic, testloader, attack, rank):
    global best_acc
    net.eval()
    net_logistic.eval()
    test_loss = 0
    correct = 0
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
        _, adv_predicted = adv_outputs.max(1)
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
            'net_logistic': net_logistic.state_dict(),
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

    # network f
    # init model and Distributed Data Parallel
    net = get_network(network=args.network, depth=args.depth, dataset=args.dataset, tran_type=args.tran_type,
                      img_size=args.img_resize, patch_size=args.patch_size, pretrain=args.pretrain)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.to(memory_format=torch.channels_last).cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=[rank])


    # network f logistic
    net_logistic = get_network(network='logistic', depth=args.depth, dataset=args.dataset, tran_type=args.tran_type,
                       img_size=args.img_resize, patch_size=args.patch_size, pretrain=args.pretrain)
    net_logistic = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_logistic)
    net_logistic = net_logistic.to(memory_format=torch.channels_last).cuda()
    net_logistic = torch.nn.parallel.DistributedDataParallel(net_logistic, device_ids=[rank], output_device=[rank])

    # upsampling for transformer
    upsample = True if args.network in transformer_list else False

    # fast dataloader
    trainloader, testloader, decoder = get_fast_dataloader(dataset=args.dataset, train_batch_size=args.batch_size,
                                                           test_batch_size=args.test_batch_size, upsample=upsample)

    # Load ADV Network
    if args.network in transformer_list:
        pretrain_ckpt_name = f'checkpoint/adv/{args.dataset}/{args.dataset}_adv_{args.network}_{args.tran_type}_patch{args.patch_size}_{args.img_resize}_best.t7'
        checkpoint = torch.load(pretrain_ckpt_name, map_location=torch.device(torch.cuda.current_device()))
    else:
        # standard
        # pretrain_ckpt_name = f'checkpoint/standard/{args.dataset}/{args.dataset}_{args.network}{args.depth}_best.t7'
        # checkpoint = torch.load(pretrain_ckpt_name, map_location=torch.device(torch.cuda.current_device()))

        # adv
        pretrain_ckpt_name = f'checkpoint/adv/{args.dataset}/{args.dataset}_adv_{args.network}{args.depth}_best.t7'
        checkpoint = torch.load(pretrain_ckpt_name, map_location=torch.device(torch.cuda.current_device()))

    net.load_state_dict(checkpoint['net'])
    rprint(f'==> {pretrain_ckpt_name}', rank)
    rprint('==> Successfully Loaded Standard checkpoint..', rank)

    # Attack loader
    if args.dataset == 'imagenet':
        rprint('Fast FGSM training', rank)
        attack = attack_loader(net=net, attack='fgsm_train', eps=args.eps/4, steps=args.steps)
    elif args.dataset == 'tiny':
        rprint('Fast FGSM training', rank)
        attack = attack_loader(net=net, attack='fgsm_train', eps=args.eps/2, steps=args.steps)
    else:
        rprint('PGD training', rank)
        attack = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps)

    # init optimizer and lr scheduler
    # optimizer network f
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=args.learning_rate,
    step_size_up=int(round(args.epochs/15))*len(trainloader),
    step_size_down=args.epochs*len(trainloader)-int(round(args.epochs/15))*len(trainloader))

    # optimizer network f logistic
    optimizer_logistic = optim.SGD(net_logistic.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler_logistic = torch.optim.lr_scheduler.CyclicLR(optimizer_logistic, base_lr=0, max_lr=args.learning_rate,
                                                      step_size_up=int(round(args.epochs / 15)) * len(trainloader),
                                                      step_size_down=args.epochs * len(trainloader) - int(
                                                          round(args.epochs / 15)) * len(trainloader))

    # optimizer network D
    params = list(net.parameters()) + list(net_logistic.parameters())
    optimizer_total = optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler_total = torch.optim.lr_scheduler.CyclicLR(optimizer_total, base_lr=0, max_lr=args.learning_rate,
                                                      step_size_up=int(round(args.epochs / 15)) * len(trainloader),
                                                      step_size_down=args.epochs * len(trainloader) - int(
                                                          round(args.epochs / 15)) * len(trainloader))

    optimizer_list = [optimizer, optimizer_logistic, optimizer_total]
    lr_scheduler_list = [lr_scheduler, lr_scheduler_logistic, lr_scheduler_total]
    scaler_list = [scaler, scaler_logistic, scaler_total]

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
        train(net, net_logistic, trainloader, optimizer_list, lr_scheduler_list, scaler_list, attack, rank)
        test(net, net_logistic, testloader, attack, rank)

def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()