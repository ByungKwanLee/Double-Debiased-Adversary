import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict

def do_freeze(net):
    for params in net.parameters():
        params.requires_grad = False

def rprint(str, rank):
    if rank==0:
        print(str)

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

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

def get_pseudo(adv_output):
    idx = adv_output.argmax(dim=1)
    b, c = adv_output.shape
    psuedo_onehot = torch.FloatTensor(b, c).cuda()
    psuedo_onehot.zero_()
    psuedo_onehot.scatter_(1, idx.unsqueeze(-1), 1)

    return psuedo_onehot

def get_onehot(adv_output, targets):
    b, c = adv_output.shape
    psuedo_onehot = torch.FloatTensor(b, c).cuda()
    psuedo_onehot.zero_()
    psuedo_onehot.scatter_(1, targets.unsqueeze(-1), 1)

    return psuedo_onehot


class SmoothCrossEntropyLoss(torch.nn.Module):
    """
    Soft cross entropy loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, reduction='mean'):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, input, target):
        num_classes = input.shape[1]
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes)
        target = (1. - self.smoothing) * target + self.smoothing / num_classes
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        loss = - (target * logprobs).sum(dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        assert False

def imshow(img, norm=False):
    img = img.cpu().numpy()
    plt.imshow(np.transpose(np.array(img / 255 if norm else img, dtype=np.float32), (1, 2, 0)))
    plt.show()

def pl(a):
    plt.plot(a.cpu())
    plt.show()

def sc(a):
    plt.scatter(range(len(a.cpu())), a.cpu(), s=2, color='darkred', alpha=0.5)
    plt.show()

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

def KLDivergence(q, p):
    kld = q * (q / p).log()
    return kld.sum(dim=1)

# attack loader
from attack.fastattack import attack_loader
from tqdm import tqdm
from torch.cuda.amp import autocast
def test_whitebox(net, testloader, attack_list, eps, rank):
    net.eval()

    attack_module = {}
    for attack_name in attack_list:
        attack_module[attack_name] = attack_loader(net=net, attack=attack_name, eps=eps, steps=30) \
                                                                                if attack_name != 'plain' else None
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

def test_blackbox(plain_net, adv_net, testloader, attack_list, eps, rank):
    plain_net.eval()
    adv_net.eval()

    attack_module = {}
    for attack_name in attack_list:
        attack_module[attack_name] = attack_loader(net=plain_net, attack=attack_name, eps=eps, steps=30)

    for key in attack_module:
        total = 0
        correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=False)
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs = attack_module[key](inputs, targets)

            with autocast():
                outputs = adv_net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[Black-Box-Test/%s] Acc: %.2f%% (%d/%d)'
                    % (key, 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

        rprint(f'{key}: {100. * correct / total:.2f}%', rank)

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


# Causal Package
def causal_loss(logits_adv, logits_inv):
    KL = lambda x, y: (x.softmax(dim=1) * (x.softmax(dim=1).log() - y.softmax(dim=1).log())).sum(dim=1)
    return (KL(logits_adv, logits_inv)).mean()
