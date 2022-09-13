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
parser.add_argument('--adv', default=True, type=bool)
parser.add_argument('--dataset', default='svhn', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=19, type=int)
parser.add_argument('--tran_type', default='small', type=str, help='tiny/small/base/large/huge')
parser.add_argument('--img_resize', default=224, type=int, help='default/224/384')
parser.add_argument('--patch_size', default=4, type=int, help='4/16/32')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--batch_size', default=128, type=float)

# attack parameters
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=8/255, type=float)
parser.add_argument('--steps', default=10, type=int)

args = parser.parse_args()

# GPU configurations
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.network in transformer_list:
    upsample = True
else:
    upsample = False

# init dataloader
_, testloader, _ = get_fast_dataloader(dataset=args.dataset, train_batch_size=1, test_batch_size=args.batch_size, dist=False, shuffle=True, upsample=upsample)

# init model
net = get_network(network=args.network, depth=args.depth, dataset=args.dataset, tran_type=args.tran_type,
                  img_size=args.img_resize, patch_size=args.patch_size, pretrain=False)

net = net.cuda()

# Load Plain Network
print('==> Loading Plain checkpoint..')
assert os.path.isdir('checkpoint/pretrain'), 'Error: no checkpoint directory found!'

# Loading checkpoint
if args.adv:
    adv_type = '_adv'
else:
    adv_type = ''

if args.network in ['vit', 'deit']:
    net_checkpoint_name = 'checkpoint/pretrain/%s/%s%s_%s_%s_patch%d_%d_best.t7' % (args.dataset, args.dataset, adv_type,
                                                                                    args.network, args.tran_type,
                                                                                    args.patch_size, args.img_resize)
elif args.network == 'swin':
    net_checkpoint_name = 'checkpoint/pretrain/%s/%s%s_%s_%s_patch%d_window7_%d_best.t7' % (args.dataset, args.dataset, adv_type,
                                                                                            args.network, args.tran_type,
                                                                                            args.patch_size, args.img_resize)
else:
    net_checkpoint_name = 'checkpoint/pretrain/%s/%s%s_%s%s_best.t7' % (args.dataset, args.dataset, adv_type, args.network, args.depth)

net_checkpoint = torch.load(net_checkpoint_name, map_location=lambda storage, loc: storage.cuda())['net']

print(" [*] Loaded network params : %s" %(net_checkpoint_name.split('/')[-1]))
checkpoint_module(net_checkpoint, net)

# init criterion
criterion = nn.CrossEntropyLoss()

def adv_test():
    net.eval()

    attack_module = {}
    # for attack_name in ['Plain', 'fgsm', 'pgd', 'cw_Linf', 'apgd', 'auto']:
    for attack_name in ['pgd']:
        args.attack = attack_name
        if args.dataset == 'imagenet' or args.dataset == 'tiny':
            attack_module[attack_name] = attack_loader(net=net, attack=args.attack, eps=2/255 if args.dataset == 'imagenet' else 4/255, steps=args.steps)
        else:
            attack_module[attack_name] = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps)

    for key in attack_module:
        total = 0
        adv_correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)

        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            adv_inputs = attack_module[key](inputs, targets)

            with autocast():
                adv_output = net(adv_inputs)
            _, adv_predicted = adv_output.max(1)

            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()

            desc = ('[Test/%s] Adv: %.4f%%' % (key, 100. * adv_correct / total))
            prog_bar.set_description(desc, refresh=True)

            # fast eval
            # if (key == 'apgd') or (key == 'auto') or (key == 'cw_Linf') or (key == 'cw'):
            #     if batch_idx >= int(len(testloader) * 0.3):
            #         break

def clean_test():
    net.eval()

    total = 0
    correct = 0
    prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)

    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        with autocast():
            cln_output = net(inputs)
        _, cln_predicted = cln_output.max(1)

        total += targets.size(0)
        correct += cln_predicted.eq(targets).sum().item()

        desc = ('[Test] Clean: %.2f%%' % (100. * correct / total))
        prog_bar.set_description(desc, refresh=True)

def class_num(dataset_name):
    if dataset_name == 'cifar10':
        return 10, 1000
    elif dataset_name == 'svhn':
        return 10, 10
    elif dataset_name == 'cifar100':
        return 100
    elif dataset_name == 'tiny':
        return 200, 50
    elif dataset_name == 'imagenet':
        return 1000, 50
    else:
        raise ValueError

def gen_drift_matrix(class_num, pred, targets, conf=False):
    confindence, predicted = pred.max(1)

    drift_matrix = torch.zeros([class_num[0], class_num[0]])
    conf_matrix = torch.zeros([class_num[0], class_num[0]])
    if conf:
        for index in range(predicted.shape[0]):
            drift_matrix[targets[index], predicted[index]] += 1
            conf_matrix[targets[index], predicted[index]] += confindence[index].cpu().detach()

        out_matrix = conf_matrix / drift_matrix

    else:
        for index in range(predicted.shape[0]):
            drift_matrix[targets[index], predicted[index]] += 1

        out_matrix = drift_matrix
    return out_matrix

def measure_adversarial_drift():
    net.eval()

    attack_module = {}
    for attack_name in ['pgd']:
        args.attack = attack_name
        if args.dataset == 'imagenet' or args.dataset == 'tiny':
            attack_module[attack_name] = attack_loader(net=net, attack=args.attack,
                                                       eps=2 / 255 if args.dataset == 'imagenet' else 4 / 255,
                                                       steps=args.steps)
        else:
            attack_module[attack_name] = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps)

    # drift matrix initialization
    drift_matrix = torch.eye(class_num(args.dataset)[0]).fill_(0)
    pred_matrix = torch.eye(class_num(args.dataset)[0]).fill_(0)

    for key in attack_module:
        total = 0
        adv_correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)

        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            adv_inputs = attack_module[key](inputs, targets)

            adv_output = net(adv_inputs)
            _, adv_predicted = adv_output.max(1)

            # drift matrix update
            #drift_matrix += gen_drift_matrix(class_num(args.dataset), adv_output, targets)
            pred_matrix += gen_drift_matrix(class_num(args.dataset), adv_output, targets, conf=True)

            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()

            desc = ('[Test/%s] Adv: %.4f%%' % (key, 100. * adv_correct / total))
            prog_bar.set_description(desc, refresh=True)

        pred_matrix = pred_matrix / (batch_idx + 1)

        print("ok")

if __name__ == '__main__':
    # set_random(777)
    clean_test()
    adv_test()
    #measure_adversarial_drift()


