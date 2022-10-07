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
parser.add_argument('--NAME', default='MART', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--port', default="12357", type=str)

# transformer parameter
parser.add_argument('--patch_size', default=16, type=int, help='4/16/32')
parser.add_argument('--img_resize', default=224, type=int, help='default/224/384')
parser.add_argument('--tran_type', default='small', type=str, help='tiny/small/base/large/huge')
parser.add_argument('--warmup-steps', default=500, type=int)
parser.add_argument("--num_steps", default=10000, type=int)
parser.add_argument('--pretrain', default=False, type=bool)

# learning parameter
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--learning_rate', default=0.05, type=float) #3e-2 for ViT
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--test_batch_size', default=64, type=float)

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
if not os.path.isdir(f'checkpoint/mart'): os.mkdir(f'checkpoint/mart')
if not os.path.isdir(f'checkpoint/mart/{args.dataset}'): os.mkdir(f'checkpoint/mart/{args.dataset}')
if args.network in transformer_list:
    saving_ckpt_name = f'./checkpoint/mart/{args.dataset}/{args.dataset}_mart_{args.network}_{args.tran_type}_patch{args.patch_size}_{args.img_resize}_best.t7'
else:
    saving_ckpt_name = f'./checkpoint/mart/{args.dataset}/{args.dataset}_mart_{args.network}{args.depth}_best.t7'


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
            loss = mart_loss(outputs, adv_outputs, targets)

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



def test(net, testloader, attack, rank):
    global best_acc
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    desc = (f'[Test/Clean] Loss: {test_loss/(0+1):.3f} | Acc: {0:.2f}')
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

    # Save clean acc.
    clean_acc = 100. * correct / total

    test_loss = 0
    correct = 0
    total = 0

    desc = (f'[Test/PGD] Loss: {test_loss / (0 + 1):.3f} | Acc: {0:.2f}')
    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = attack(inputs, targets)

        # Accerlating forward propagation
        with autocast():
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = (f'[Test/PGD] Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}')
        prog_bar.set_description(desc, refresh=True)

    # Save adv acc.
    adv_acc = 100. * correct / total

    # compute acc
    acc = (clean_acc + adv_acc)/2

    rprint(f'Current Accuracy is {clean_acc:.2f}/{adv_acc:.2f}!!', rank)

    if acc > best_acc:
        state = {
            'net': net.state_dict(),
        }

        torch.save(state, saving_ckpt_name)
        rprint(f'Saving~ {saving_ckpt_name}', rank)

        # update best acc
        best_acc = acc

def mart_loss(logits,
            logits_adv,
            targets):
    kl = torch.nn.KLDivLoss(reduction='none')
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == targets, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, targets) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (targets.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / logits.shape[0]) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(3) * loss_robust
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
                      dataset=args.dataset)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.to(memory_format=torch.channels_last).cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=[rank])

    # fast init dataloader
    trainloader, testloader, decoder = get_fast_dataloader(dataset=args.dataset,
                                                  train_batch_size=args.batch_size,
                                                  test_batch_size=args.test_batch_size)

    # Load ADV Network
    if args.network in transformer_list:
        pretrain_ckpt_name = f'checkpoint/adv/{args.dataset}/{args.dataset}_adv_{args.network}_{args.tran_type}_patch{args.patch_size}_{args.img_resize}_best.t7'
        checkpoint = torch.load(pretrain_ckpt_name, map_location=torch.device(torch.cuda.current_device()))
    else:
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
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=args.learning_rate,
    step_size_up=args.epoch * len(trainloader)/2, step_size_down=args.epoch * len(trainloader)/2)

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