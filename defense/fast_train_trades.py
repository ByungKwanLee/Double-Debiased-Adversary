# Import built-in module
import argparse
import warnings
warnings.filterwarnings(action='ignore')

# Import torch
import torch.optim as optim
import torch.distributed as dist

# Import Custom Utils
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *
from utils.scheduler import WarmupCosineSchedule

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
parser.add_argument('--NAME', default='TRADES', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--port', default="12000", type=str)

# transformer parameter
parser.add_argument('--patch_size', default=16, type=int, help='16')
parser.add_argument('--img_resize', default=224, type=int, help='224')
parser.add_argument('--tran_type', default='small', type=str, help='small/base')
parser.add_argument('--warmup-steps', default=500, type=int)
parser.add_argument("--num_steps", default=10000, type=int)
parser.add_argument('--pretrain', default=True, type=bool)

# learning parameter
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--learning_rate', default=0.5, type=float) #3e-2 for ViT # 0.01
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--test_batch_size', default=128, type=float)

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

# make checkpoint folder and set checkpoint name for saving
if not os.path.isdir(f'checkpoint'): os.mkdir(f'checkpoint')
if not os.path.isdir(f'checkpoint/trades'): os.mkdir(f'checkpoint/trades')
if not os.path.isdir(f'checkpoint/trades/{args.dataset}'): os.mkdir(f'checkpoint/trades/{args.dataset}')
if args.network in transformer_list:
    saving_ckpt_name = f'./checkpoint/trades/{args.dataset}/{args.dataset}_trades_{args.network}_{args.tran_type}_patch{args.patch_size}_{args.img_resize}_best.t7'
else:
    saving_ckpt_name = f'./checkpoint/trades/{args.dataset}/{args.dataset}_trades_{args.network}{args.depth}_best.t7'

def train(net, trainloader, optimizer, lr_scheduler, scaler, attack):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    desc = (f'[Train/LR={lr_scheduler.get_lr()[0]:.3f}] Loss: {0:.3f} | Acc: {0:.2f}')

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inputs = attack(inputs, targets)

        # Accerlating forward propagation
        optimizer.zero_grad()
        with autocast():
            outputs = net(inputs)
            adv_outputs = net(adv_inputs)
            loss = trades_loss(outputs, adv_outputs, targets)

        # Accerlating backward propagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # scheduling for Cyclic LR
        lr_scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = (f'[Train/LR={lr_scheduler.get_lr()[0]:.3f}] Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}')
        prog_bar.set_description(desc, refresh=True)



def test(net, testloader, attack, rank):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = (f'[Test/Clean] Loss: {test_loss/(0+1):.3f} | Acc: {0:.2f}%')

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=False)
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

        desc = (f'[Test/Clean] Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}%')
        prog_bar.set_description(desc, refresh=True)

    # Save clean acc.
    clean_acc = 100. * correct / total

    test_loss = 0
    correct = 0
    total = 0

    desc = (f'[Test/PGD] Loss: {test_loss / (0 + 1):.3f} | Acc: {0:.2f}%')
    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs = attack(inputs, targets)
        inputs, targets = inputs.cuda(), targets.cuda()

        # Accerlating forward propagation
        with autocast():
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = (f'[Test/PGD] Loss: {test_loss/(batch_idx + 1):.3f} | Acc: {100.*correct/total:.2f}%')
        prog_bar.set_description(desc, refresh=True)

    # Save adv acc.
    adv_acc = 100. * correct / total

    # compute acc
    acc = (clean_acc + adv_acc)/2

    # current accuracy print
    rprint(f'Current Accuracy is {clean_acc:.2f}/{adv_acc:.2f}!!', rank)

    if acc > best_acc:
        state = {
            'net': net.state_dict(),
        }

        torch.save(state, saving_ckpt_name)
        rprint(f'Saving~ {saving_ckpt_name}', rank)

        # update best acc
        best_acc = acc

# for trade loss with no modification
def trades_loss(logits,
                logits_adv,
                targets):
    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    loss_natural = F.cross_entropy(logits, targets)
    loss_robust = (1.0 / logits.shape[0]) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
    loss = loss_natural + float(2) * loss_robust
    return loss


def main_worker(rank, ngpus_per_node=ngpus_per_node):

    # print configuration
    print_configuration(args, rank)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # DDP environment settings
    print(f'Use GPU: {gpu_list[rank]} for training')
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)

    # init model and Distributed Data Parallel
    net = get_network(network=args.network,
                      depth=args.depth,
                      dataset=args.dataset,
                      tran_type=args.tran_type,
                      img_size=args.img_resize,
                      patch_size=args.patch_size,
                      pretrain=args.pretrain)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.to(memory_format=torch.channels_last).cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=[rank])

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
        rprint('PGD and FGSM MIX training', rank)
        pgd_attack = attack_loader(net=net, attack='pgd', eps=args.eps/2, steps=args.steps)
        fgsm_attack = attack_loader(net=net, attack='fgsm_train', eps=args.eps/2, steps=args.steps)
        attack = MixAttack(net=net, slowattack=pgd_attack, fastattack=fgsm_attack, train_iters=len(trainloader))
    else:
        rprint('PGD training', rank)
        attack = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps)

    # init optimizer and lr scheduler
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    # init optimizer and lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=args.learning_rate,
    step_size_up=int(round(args.epochs/15))*len(trainloader),
    step_size_down=args.epochs*len(trainloader)-int(round(args.epochs/15))*len(trainloader))


    # training and testing
    for epoch in range(args.epochs):
        rprint(f'\nEpoch: {epoch + 1}', rank)
        if args.dataset == "imagenet":
            if args.network in transformer_list:
                res = args.img_resize
            else:
                res = get_resolution(epoch=epoch, min_res=160, max_res=192,
                                     start_ramp=int(math.floor(args.epochs * 0.5)),
                                     end_ramp=int(math.floor(args.epochs * 0.7)))
            decoder.output_size = (res, res)
        train(net, trainloader, optimizer, lr_scheduler, scaler, attack)
        test(net, testloader, attack, rank)

def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()