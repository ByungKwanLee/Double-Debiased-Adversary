from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# attack loader
from attack.fastattack import attack_loader

# warning ignore
import warnings
warnings.filterwarnings("ignore")
from utils.utils import str2bool

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--base', default='awp', type=str)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--gpu', default='1', type=str)

# transformer parameter
parser.add_argument('--tran_type', default='small', type=str, help='tiny/small/base/large/huge')
parser.add_argument('--img_resize', default=224, type=int, help='default/224/384')
parser.add_argument('--patch_size', default=16, type=int, help='4/16/32')

# attack parameters
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=8/255, type=float)
parser.add_argument('--steps', default=10, type=int)

args = parser.parse_args()

# print configuration
print_configuration(args, 0)

# GPU configurations
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.cuda.set_device(f'cuda:{args.gpu}')

# upsampling for transformer
upsample = True if args.network in transformer_list else False

# init dataloader
_, testloader, _ = get_fast_dataloader(dataset=args.dataset, train_batch_size=1, test_batch_size=args.batch_size, dist=False, upsample=upsample)

# init model F
netF = get_network(network=args.network, depth=args.depth, dataset=args.dataset, tran_type=args.tran_type,
                  img_size=args.img_resize, patch_size=args.patch_size, pretrain=False)
netF = netF.cuda()


# checkpoint base tag
base_tag = '' if args.base == 'standard' else '_' + args.base
# setting checkpoint name
if args.network in transformer_list:
    net_checkpoint_name = f'checkpoint/{args.base}/{args.dataset}/{args.dataset}{base_tag}_{args.network}_{args.tran_type}_patch{args.patch_size}_{args.img_resize}_best.t7'
else:
    net_checkpoint_name = f'checkpoint/{args.base}/{args.dataset}/{args.dataset}{base_tag}_{args.network}{args.depth}_best.t7'

# load checkpoint
netF_checkpoint = torch.load(net_checkpoint_name, map_location=lambda storage, loc: storage.cuda())['net']
checkpoint_module(netF_checkpoint, netF)
print(" [*] Loaded network params : %s" %(net_checkpoint_name.split('/')[-1]))

# init criterion
criterion = nn.CrossEntropyLoss()


def daml_test():
    netF.eval()

    attack_module = {}
    # for attack_name in ['Plain', 'fgsm', 'pgd', 'cw_Linf', 'apgd', 'auto']:
    for attack_name in ['pgd']:
        args.attack = attack_name
        if args.dataset == 'imagenet':
            attack_module[attack_name] = attack_loader(net=netF, attack=args.attack, eps=args.eps/4, steps=args.steps)
        elif args.dataset == 'tiny':
            attack_module[attack_name] = attack_loader(net=netF, attack=args.attack, eps=args.eps/2, steps=args.steps)
        else:
            attack_module[attack_name] = attack_loader(net=netF, attack=args.attack, eps=args.eps, steps=args.steps)

    for key in attack_module:
        total = 0
        correctF = 0
        adv_correctF = 0
        correctG = 0
        correctFG = 0
        correct_want1 = 0
        correct_want2 = 0
        correct_want3 = 0
        correct_want4 = 0

        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)

        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            adv_inputs = attack_module[key](inputs, targets)

            with autocast():
                adv_outputsF = netF(adv_inputs)
                outputsF = netF(inputs)

            _, adv_predictedF = adv_outputsF.max(1)
            _, predictedF = outputsF.max(1)

            want1 = (predictedF == targets) * (adv_predictedF == targets)
            want2 = (predictedF != targets) * (adv_predictedF == targets)
            want3 = (predictedF == targets) * (adv_predictedF != targets)
            want4 = (predictedF != targets) * (adv_predictedF != targets)



            total += targets.size(0)
            correctF += predictedF.eq(targets).sum().item()
            adv_correctF += adv_predictedF.eq(targets).sum().item()

            correct_want1 += want1.sum().item()
            correct_want2 += want2.sum().item()
            correct_want3 += want3.sum().item()
            correct_want4 += want4.sum().item()

            desc = (f'[Test/{key}] Clean: {100.*correctF/total:.2f}% | Adv: {100.*adv_correctF/total:.2f}% | G ACC: {100.*correctG/total:.2f}% | FG similarity: {100.*correctFG/total:.2f}%')
            prog_bar.set_description(desc, refresh=True)

        print('------------------------')
        print('(predictedF == targets) * (adv_predictedF == targets)')
        print(100. * correct_want1 / total)
        print('(predictedF != targets) * (adv_predictedF == targets)')
        print(100. * correct_want2 / total)
        print('(predictedF == targets) * (adv_predictedF != targets)')
        print(100. * correct_want3 / total)
        print('(predictedF != targets) * (adv_predictedF != targets)')
        print(100. * correct_want4 / total)
        print('------------------------')

        # fast eval
        if args.dataset == 'imagenet':
            if batch_idx >= int(len(testloader) * 0.2):
                break


def class_num(dataset_name):
    if dataset_name == 'cifar10':
        return 10, 1000
    elif dataset_name == 'cifar100':
        return 100
    elif dataset_name == 'tiny':
        return 200, 50
    elif dataset_name == 'imagenet':
        return 1000, 50
    else:
        raise ValueError


if __name__ == '__main__':
    daml_test()


