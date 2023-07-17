import os
import math
import torch
import random
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

transformer_list = ['vit', 'deit', 'swin', 'cait', 'tnt']

def rprint(str, rank):
    if rank==0:
        print(str)

def checkpoint_module(checkpoint, net):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_onehot(adv_output, targets):
    b, c = adv_output.shape
    psuedo_onehot = torch.FloatTensor(b, c).cuda()
    psuedo_onehot.zero_()
    psuedo_onehot.scatter_(1, targets.unsqueeze(-1), 1)

    return psuedo_onehot

def get_resolution(epoch, min_res, max_res, end_ramp, start_ramp):
    assert min_res <= max_res

    if epoch <= start_ramp:
        return min_res

    if epoch >= end_ramp:
        return max_res

    # otherwise, linearly interpolate to the nearest multiple of 32
    interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
    final_res = int(np.round(interp[0] / 32)) * 32
    return final_res

def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def print_configuration(args, rank):
    dict = vars(args)
    if rank == 0:
        print('------------------Configurations------------------')
        for key in dict.keys():
            print("{}: {}".format(key, dict[key]))
        print('-------------------------------------------------')

# Mixattack for Tiny-ImageNet
class MixAttack(object):
    def __init__(self, net, slowattack, fastattack, train_iters):
        self.net = net
        self.slowattack = slowattack
        self.fastattack = fastattack
        self.train_iters = train_iters
        self.ratio = 0.3
        self.current_iter = 0

    def __call__(self, inputs, targets):
        # training
        if self.net.training:
            adv_inputs = self.slowattack(inputs, targets) \
                if self._iter < self.train_iters * self.ratio else self.fastattack(inputs, targets)
            self.iter()
            self.check()
        # testing
        else:
            adv_inputs = self.fastattack(inputs, targets)
        return adv_inputs

    def iter(self):
        self.current_iter = self.current_iter+1

    def check(self):
        if self.train_iters == self.current_iter:
            self.current_iter = 0

    @property
    def _iter(self):
        return self.current_iter


# attack loader
from attack.fastattack import attack_loader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
def test_whitebox(net, dataset, testloader, attack_list, steps, eps, rank):
    net.eval()

    attack_module = {}
    for attack_name in attack_list:
        if dataset == 'imagenet':
            attack_module[attack_name] = attack_loader(net=net, attack=attack_name, eps=eps/4, steps=steps) if attack_name != 'plain' else None
        elif dataset == 'tiny':
            attack_module[attack_name] = attack_loader(net=net, attack=attack_name, eps=eps/2, steps=steps) if attack_name != 'plain' else None
        else:
            attack_module[attack_name] = attack_loader(net=net, attack=attack_name, eps=eps, steps=steps) if attack_name != 'plain' else None

    for key in attack_module:
        total = 0
        correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=False)
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()

            if key != 'plain':
                inputs = attack_module[key](inputs, targets)
            with autocast():
                outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[White-Box-Test/%s] Acc: %.2f%% (%d/%d)'
                    % (key, 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

        rprint(f'{key}: {100. * correct / total:.2f}%', rank)

# awp package
EPS = 1E-20
def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])

class AdvWeightPerturb(object):
    def __init__(self, model, proxy, lr, gamma, autocast, GradScaler):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.gamma = gamma
        self.proxy_optim = torch.optim.SGD(proxy.parameters(), lr=lr)
        self.autocast = autocast
        self.scaler = GradScaler()

    def calc_awp(self, adv_inputs, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        self.proxy_optim.zero_grad()
        with self.autocast():
            loss = -F.cross_entropy(self.proxy(adv_inputs), targets)

        # Accelerating backward propagation
        self.scaler.scale(loss).backward()
        self.scaler.step(self.proxy_optim)
        self.scaler.update()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


def dml_loss(pred, adv_pred, targets):
    is_attack = adv_pred.max(1)[1] != targets
    adv_prob = adv_pred.softmax(dim=1)
    tar_prob = adv_prob[is_attack] * get_onehot(adv_pred[is_attack], targets[is_attack])
    ntar_prob = adv_prob[is_attack] * get_onehot(adv_pred[is_attack], adv_pred.max(1)[1][is_attack])
    return ((1/ntar_prob.sum(dim=1).detach()-1) * -(tar_prob.sum(dim=1)+1e-3).log()).mean() \
            + ((1/ntar_prob.sum(dim=1).detach()-1) * (ntar_prob.sum(dim=1)+1e-3).log()).mean() \
            + F.cross_entropy(pred, targets)




