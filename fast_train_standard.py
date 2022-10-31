# Future
from __future__ import print_function

# Built-in module
import argparse
import warnings
warnings.filterwarnings(action='ignore')

# torch pkg
import torch
import torch.optim as optim
import torch.distributed as dist

# Cudnn settings
torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

# Import Custom Utils
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *
from utils.scheduler import WarmupCosineSchedule

# Accelerating forward and backward
from torch.cuda.amp import GradScaler, autocast

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--NAME', default='STANDARD', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='swin', type=str)
parser.add_argument('--depth', default=12, type=int, help='cait depth = 24')
parser.add_argument('--gpu', default='0,1,2', type=str)
parser.add_argument('--port', default="12359", type=str)

# transformer parameter
parser.add_argument('--tran_type', default='small', type=str, help='tiny/small/base/large/huge//xxs/s')
parser.add_argument('--img_resize', default=224, type=int, help='32/224')
parser.add_argument('--patch_size', default=4, type=int, help='4/16')
parser.add_argument('--warmup-steps', default=500, type=int)
parser.add_argument("--num_steps", default=10000, type=int)
parser.add_argument('--pretrain', default=True, type=bool)

# learning parameter
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float) #1e-4 for ViT
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--test_batch_size', default=64, type=float)

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
if not os.path.isdir(f'checkpoint/standard'): os.mkdir(f'checkpoint/standard')
if not os.path.isdir(f'checkpoint/standard/{args.dataset}'): os.mkdir(f'checkpoint/standard/{args.dataset}')
if args.network in transformer_list:
    saving_ckpt_name = f'./checkpoint/standard/{args.dataset}/{args.dataset}_{args.network}_{args.tran_type}_patch{args.patch_size}_{args.img_resize}_best.t7'
else:
    saving_ckpt_name = f'./checkpoint/standard/{args.dataset}/{args.dataset}_{args.network}{args.depth}_best.t7'

def train(net, trainloader, optimizer, lr_scheduler, scaler):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    desc = (f'[Train/LR={lr_scheduler.get_lr()[0]:.3f}] Loss: {0:.3f} | Acc: {0:.2f}')
    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()

        # Accerlating forward propagation
        optimizer.zero_grad()
        with autocast():
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

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

        desc = (f'[Train/LR={lr_scheduler.get_lr()[0]:.3f}] Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}')

        prog_bar.set_description(desc, refresh=True)

def test(net, testloader, rank):
    global best_acc

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    desc = (f'[Test/Clean] Loss: {test_loss / (0 + 1):.3f} | Acc: {0:.2f}')
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

        desc = (f'[Test/Clean] Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}')
        prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100.*correct/total

    rprint('Current Accuracy is {:.2f}!!'.format(acc), rank)

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

    if args.network == 'tnt' and args.tran_type == 'base':
        pretrain_ckpt_name = 'checkpoint/standard/imagenet/imagenet_tnt_base_patch16_224_best.pth.tar'
        checkpoint = torch.load(pretrain_ckpt_name, map_location=torch.device(torch.cuda.current_device()))
        net.load_state_dict(checkpoint, strict=False)
        rprint(f'==> {pretrain_ckpt_name}', rank)
        rprint('==> Successfully Loaded Standard checkpoint..', rank)

    if args.network in transformer_list:
        t_total = args.num_steps
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=args.learning_rate,
        step_size_up=int(round(args.epochs/15))*len(trainloader),
        step_size_down=args.epochs*len(trainloader)-int(round(args.epochs/15))*len(trainloader))

    # training and testing
    for epoch in range(args.epochs):
        rprint('\nEpoch: %d' % (epoch+1), rank)
        if args.dataset == "imagenet":
            if args.network in transformer_list:
                res = args.img_resize
            else:
                res = get_resolution(epoch=epoch, min_res=160, max_res=192,
                                     start_ramp=int(math.floor(args.epochs * 0.5)),
                                     end_ramp=int(math.floor(args.epochs * 0.7)))
            decoder.output_size = (res, res)
        train(net, trainloader, optimizer, lr_scheduler, scaler)
        test(net, testloader, rank)

def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()