import time
from torchattacks.attack import Attack
from attack.libfastattack.FastAPGD import FastAPGD
from attack.libfastattack.FastFAB import FastFAB
from attack.libfastattack.FastMultiAttack import FastMultiAttack


class FastAutoAttack(Attack):
    def __init__(self, model, eps=.3, steps=30, alpha_max=0.1,  eta=1.05, beta=0.9, seed=None):
        super().__init__("FastAutoAttack", model)
        self.eps = eps
        self.seed = seed
        self._supported_mode = ['default']

        self.autoattack = FastMultiAttack([
            FastAPGD(model, eps=eps, seed=self.get_seed(), loss='ce', steps=steps),
            FastFAB(model, eps=eps, seed=self.get_seed(), alpha_max=alpha_max, gamma=0.2, eta=eta, beta=beta),
        ])


    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return time.time() if self.seed is None else self.seed