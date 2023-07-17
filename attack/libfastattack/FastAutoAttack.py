import time
from torchattacks.attack import Attack
from attack.libfastattack.FastAPGD import FastAPGD
from attack.libfastattack.FastAPGDT import FastAPGDT
from attack.libfastattack.FastSquare import FastSquare
from attack.libfastattack.FastFAB import FastFAB
from attack.libfastattack.FastMultiAttack import FastMultiAttack


class FastAutoAttack(Attack):
    def __init__(self, model, steps, version, eps=.3, seed=None):
        super().__init__("FastAutoAttack", model)
        self.eps = eps
        self.seed = seed
        self._supported_mode = ['default']

        if version == "standard":
            # standard version
            self.autoattack = FastMultiAttack([
                FastAPGD(model=model, eps=eps, seed=self.get_seed(), loss='ce', steps=steps),
                FastAPGDT(model=model, eps=eps, seed=self.get_seed(), steps=steps),
                FastFAB(model, eps=eps, seed=self.get_seed(), multi_targeted=True),
                FastSquare(model, eps=eps, seed=self.get_seed())
            ])
        elif version == "plus":
            # plus version
            self.autoattack = FastMultiAttack([
                FastAPGD(model=model, eps=eps, seed=self.get_seed(), loss='ce', steps=steps),
                FastAPGD(model=model, eps=eps, seed=self.get_seed(), loss='dlr', steps=steps),
                FastFAB(model, eps=eps, seed=self.get_seed()),
                FastSquare(model, eps=eps, seed=self.get_seed()),
                FastAPGDT(model=model, eps=eps, seed=self.get_seed(), steps=steps),
                FastFAB(model, eps=eps, seed=self.get_seed(), multi_targeted=True)
            ])
        elif  version == "rand":
            # rand version
            self.autoattack = FastMultiAttack([
                FastAPGD(model=model, eps=eps, seed=self.get_seed(), loss='ce', steps=steps),
                FastAPGD(model=model, eps=eps, seed=self.get_seed(), loss='dlr', steps=steps),
            ])
        elif  version == "custom":
            # custom version
            self.autoattack = FastMultiAttack([
                FastAPGD(model=model, eps=eps, seed=self.get_seed(), loss='ce', steps=steps),
                FastFAB(model, eps=eps, seed=self.get_seed()),
            ])
        else:
            raise ValueError


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