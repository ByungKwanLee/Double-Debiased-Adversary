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
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--port', default="12355", type=str)

# transformer parameter
parser.add_argument('--patch_size', default=16, type=int, help='4/16/32')
parser.add_argument('--img_resize', default=224, type=int, help='32/224')
parser.add_argument('--tran_type', default='base', type=str, help='tiny/small/base/large/huge')
parser.add_argument('--warmup-steps', default=500, type=int)
parser.add_argument("--num_steps", default=10000, type=int)

# learning parameter
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--learning_rate', default=0.5, type=float) #3e-2 for ViT
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
scalerF = GradScaler()
scalerG = GradScaler()
scalerD = GradScaler()

def train(netF, netG, trainloader, optimizer, lr_scheduler, scaler, attack):
    optimizerF, optimizerG, optimizerD = optimizer
    lr_schedulerF, lr_schedulerG, lr_schedulerD = lr_scheduler
    scalerF, scalerG, scalerD = scaler

    netF.train()
    netG.train()
    train_lossF, train_lossG, train_lossD = 0, 0, 0
    correctF, correctG, correct = 0, 0, 0
    total = 0

    desc = ('[Tr/lrFG=%.3f/%.3f] Loss: (F) %.3f | (G) %.3f | (D) %.3f | Acc: (F) %.2f%% | (G) %.2f%% | (F==G) %.2f%%' %
            (lr_schedulerF.get_lr()[0], lr_schedulerG.get_lr()[0], 0, 0, 0, 0, 0, 0))

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)

    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inputs = attack(inputs, targets)

        # sample splitting
        inputs1, inputs2 = inputs.split(args.batch_size//2)
        adv_inputs1, adv_inputs2 = adv_inputs.split(args.batch_size // 2)
        targets1, targets2 = targets.split(args.batch_size//2)


        # DAML STEP [1]
        # (1-1): optimizerF init
        optimizerF.zero_grad()
        with autocast():
            adv_outputsF = netF(adv_inputs1)
            lossF = F.cross_entropy(adv_outputsF, targets1)
        scalerF.scale(lossF).backward()
        scalerF.step(optimizerF)
        scalerF.update()

        # (1-2): optimizerG init
        optimizerG.zero_grad()
        with autocast():
            adv_outputsF = netF(adv_inputs1)
            outputsF = netF(inputs1)
            outputsG = netG(inputs1)

            # training accuracy, vgg-16, cifar10, clean input만 보고 미래에 T가 들어왔을 때 50% 로 어디로 갈지 예측가능
            # first lossG: logit difference
            diff_y = adv_outputsF - outputsF
            lossG = (outputsG - diff_y.detach()).square().mean()

            # training accuracy, vgg-16, cifar10, clean input만 보고 미래에 T가 들어왔을 때 21% 로 어디로 갈지 예측가능
            # second lossG: KL divergence
            # pred_prob = (outputsF.detach() + outputsG).softmax(dim=1)
            # tar_prob = adv_outputsF.softmax(dim=1).detach()
            # lossG = (pred_prob * ((pred_prob+1e-3).log() - (tar_prob+1e-3).log())).sum(dim=1).mean()

            # The best training accuracy, vgg-16, cifar10, clean input만 보고 미래에 T가 들어왔을 때 44% 로 어디로 갈지 예측가능
            # third lossG: Split Correct prediction and Wrong prediction
            # _, attack_targets = adv_outputsF.max(1)
            # lossG = F.cross_entropy(outputsG, attack_targets)

            # training accuracy, vgg-16, cifar10, clean input만 보고 미래에 T가 들어왔을 때 41% 로 어디로 갈지 예측가능
            # fourth lossG: Plus version of Split Correct prediction and Wrong prediction
            # pred_prob = outputsG.softmax(dim=1)
            # tar_prob = adv_outputsF.softmax(dim=1).detach()
            # _, attack_targets = adv_outputsF.max(1)
            # lossG = F.cross_entropy(outputsG, attack_targets) + (pred_prob * ((pred_prob+1e-3).log() - (tar_prob+1e-3).log())).sum(dim=1).mean()

        scalerG.scale(lossG).backward()
        scalerG.step(optimizerG)
        scalerG.update()

        # DAML STEP [2]
        # (2): optimizerD init
        # optimizerD.zero_grad()
        # with autocast():
        #     outputsF = netF(inputs2)
        #     outputsG = netG(inputs2)
        #     lossD = F.cross_entropy(outputsF + outputsG.detach(), targets2)
        #
        # scalerD.scale(lossD).backward()
        # scalerD.step(optimizerF)
        # scalerD.update()

        # scheduling for Cyclic LR
        lr_schedulerF.step()
        lr_schedulerG.step()
        # lr_schedulerD.step()

        # train_lossF += lossF.item()
        train_lossG += lossG.item()
        # train_lossD += lossD.item()

        _, predictedF = adv_outputsF.max(1)
        _, predictedG = (outputsG+outputsF).max(1)

        total += targets1.size(0)
        correctF += predictedF.eq(targets1).sum().item()
        correctG += predictedG.eq(targets1).sum().item()
        correct += predictedG.eq(predictedF).sum().item()

        desc = ('[Tr/lrFG=%.3f/%.3f] Loss: (F) %.3f | (G) %.3f | (D) %.3f | Acc: (F) %.2f%% | (G) %.2f%% | (F==G) %.2f%%' %
                (lr_schedulerF.get_lr()[0], lr_schedulerG.get_lr()[0], train_lossF / (batch_idx + 1), train_lossG / (batch_idx + 1),
                 train_lossD / (batch_idx + 1), 100. * correctF / total, 100. * correctG / total, 100. * correct / total))
        prog_bar.set_description(desc, refresh=True)

def test(netF, netG, testloader, attack, rank):
    global best_acc
    netF.eval()
    netG.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[Test/Clean] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()

        # Accerlating forward propagation
        with autocast():
            outputs = netF(inputs)
            loss = F.cross_entropy(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[Test/Clean] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    # Save clean acc.
    clean_acc = 100. * correct / total

    test_loss = 0
    correct = 0
    correctG = 0
    total = 0

    desc = ('[Test/PGD] Loss: %.3f | Acc: (F) %.3f%% | (F==G) %.3f%%'
            % (test_loss / (0 + 1), 0, 0))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inputs = attack(inputs, targets)

        # Accerlating forward propagation
        with autocast():
            adv_outputs = netF(adv_inputs)
            outputsF = netF(inputs)
            outputsG = netG(inputs)
            loss = F.cross_entropy(adv_outputs, targets)

        test_loss += loss.item()
        _, predicted = adv_outputs.max(1)
        _, predictedG = (outputsG+outputsF).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        correctG += predictedG.eq(predicted).sum().item()

        desc = ('[Test/PGD] Loss: %.3f | Acc: (F) %.3f%% | (F==G) %.3f%%'
                % (test_loss / (batch_idx + 1), 100. * correct / total, 100 * correctG / total))
        prog_bar.set_description(desc, refresh=True)

    # Save adv acc.
    adv_acc = 100. * correct / total

    # compute acc
    acc = 100 * correctG / total

    rprint('Current Accuracy is {:.2f}/{:.2f}/{:.2f}!!'.format(clean_acc, adv_acc, 100 * correctG / total), rank)

    state = {
        'net': netF.state_dict(),
        'netG': netG.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('checkpoint/pretrain'):
        os.mkdir('checkpoint/pretrain')

    if rank == 0:
        if args.network in transformer_list:
            torch.save(state, './checkpoint/pretrain/%s/%s_daml_%s_%s_patch%d_%d_best.t7' % (args.dataset, args.dataset,
                                                                                            args.network, args.tran_type,
                                                                                            args.patch_size, args.img_resize))
            print('Saving~ ./checkpoint/pretrain/%s/%s_daml_%s_%s_patch%d_%d_best.t7' % (args.dataset, args.dataset,
                                                                                        args.network, args.tran_type,
                                                                                        args.patch_size, args.img_resize))
        elif args.network == 'swin':
            torch.save(state, './checkpoint/pretrain/%s/%s_daml_%s_%s_patch%d_window7_%d_best.t7' % (args.dataset, args.dataset,
                                                                                                    args.network, args.tran_type,
                                                                                                    args.patch_size, args.img_resize))
            print('Saving~ ./checkpoint/pretrain/%s/%s_daml_%s_%s_patch%d_window7_%d_best.t7' % (args.dataset, args.dataset,
                                                                                                args.network, args.tran_type,
                                                                                                args.patch_size, args.img_resize))
        else:
            torch.save(state, './checkpoint/pretrain/%s/%s_daml_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                                args.network, args.depth))
            print('Saving~ ./checkpoint/pretrain/%s/%s_daml_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                            args.network, args.depth))

def main_worker(rank, ngpus_per_node=ngpus_per_node):
    # print configuration
    print_configuration(args, rank)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # DDP environment settings
    print("Use GPU: {} for training".format(gpu_list[rank]))
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)

    # init model and Distributed Data Parallel
    netF = get_network(network=args.network, depth=args.depth, dataset=args.dataset, tran_type=args.tran_type,
                      img_size=args.img_resize, patch_size=args.patch_size, pretrain=args.pretrain)
    netF = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netF)
    netF = netF.to(memory_format=torch.channels_last).cuda()
    netF = torch.nn.parallel.DistributedDataParallel(netF, device_ids=[rank], output_device=[rank])

    netG = get_network(network=args.network, depth=args.depth, dataset=args.dataset, tran_type=args.tran_type,
                       img_size=args.img_resize, patch_size=args.patch_size, pretrain=args.pretrain)
    netG.apply(weights_init)
    netG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netG)
    netG = netG.to(memory_format=torch.channels_last).cuda()
    netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=[rank], output_device=[rank])

    # upsampling for transformer
    upsample = True if args.network in transformer_list else False

    # fast dataloader
    trainloader, testloader, decoder = get_fast_dataloader(dataset=args.dataset, train_batch_size=args.batch_size,
                                                           test_batch_size=args.test_batch_size, upsample=upsample)

    # Load Plain Network
    if args.network in transformer_list:
        checkpoint_name = 'checkpoint/pretrain/%s/%s_adv_%s_%s_patch%d_%d_best.t7' % (args.dataset, args.dataset,
                                                                                      args.network, args.tran_type,
                                                                                      args.patch_size, args.img_resize)
        checkpoint = torch.load(checkpoint_name, map_location=torch.device(torch.cuda.current_device()))
    elif args.network == 'swin':
        checkpoint_name = 'checkpoint/pretrain/%s/%s_adv_%s_%s_patch%d_window7_%d_best.t7' % (args.dataset, args.dataset,
                                                                                              args.network, args.tran_type,
                                                                                              args.patch_size, args.img_resize)
        checkpoint = torch.load(checkpoint_name, map_location=torch.device(torch.cuda.current_device()))
    else:
        checkpoint_name = 'checkpoint/pretrain/%s/%s_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)
        checkpoint = torch.load(checkpoint_name, map_location=torch.device(torch.cuda.current_device()))

    netF.load_state_dict(checkpoint['net'])
    rprint(f'==> {checkpoint_name}', rank)
    rprint('==> Successfully Loaded Standard checkpoint..', rank)

    # Attack loader
    if args.dataset == 'imagenet':
        rprint('Fast FGSM training', rank)
        attack = attack_loader(net=netF, attack='fgsm_train', eps=args.eps/4, steps=args.steps)
    elif args.dataset == 'tiny':
        rprint('Fast FGSM training', rank)
        attack = attack_loader(net=netF, attack='fgsm_train', eps=args.eps/2, steps=args.steps)
    else:
        rprint('PGD training', rank)
        attack = attack_loader(net=netF, attack=args.attack, eps=args.eps, steps=args.steps)

    # init optimizer and lr scheduler
    if args.network in transformer_list:
        t_total = args.num_steps
        optimizerF = optim.SGD(netF.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        lr_schedulerF = WarmupCosineSchedule(optimizerF, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        optimizerF = optim.SGD(netF.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        lr_schedulerF = torch.optim.lr_scheduler.CyclicLR(optimizerF, base_lr=0, max_lr=args.learning_rate,
        step_size_up=int(round(args.epochs/15))*len(trainloader),
        step_size_down=args.epochs*len(trainloader)-int(round(args.epochs/15))*len(trainloader))

    optimizerG = optim.SGD(netG.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_schedulerG = torch.optim.lr_scheduler.CyclicLR(optimizerG, base_lr=0, max_lr=args.learning_rate,
                                                      step_size_up=int(round(args.epochs / 15)) * len(trainloader),
                                                      step_size_down=args.epochs * len(trainloader) - int(
                                                          round(args.epochs / 15)) * len(trainloader))

    params = list(netF.parameters()) + list(netG.parameters())
    optimizerD = optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_schedulerD = torch.optim.lr_scheduler.CyclicLR(optimizerD, base_lr=0, max_lr=args.learning_rate,
                                                      step_size_up=int(round(args.epochs / 15)) * len(trainloader),
                                                      step_size_down=args.epochs * len(trainloader) - int(
                                                          round(args.epochs / 15)) * len(trainloader))

    optimizer = [optimizerF, optimizerG, optimizerD]
    lr_scheduler = [lr_schedulerF, lr_schedulerG, lr_schedulerD]
    scaler = [scalerF, scalerG, scalerD]

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
        train(netF, netG, trainloader, optimizer, lr_scheduler, scaler, attack)
        test(netF, netG, testloader, attack, rank)

def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()