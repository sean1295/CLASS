import torch

class Normalizer:
    """
    parent class, subclass by defining the `normalize` and `unnormalize` methods
    """

    def __init__(self, X):
        self.X = X
        self.mins = self.X.min(dim=0)[0]
        self.maxs = self.X.max(dim=0)[0]
        self.device = X.device

    def __repr__(self):
        return (
            f"[ Normalizer ] dim: {self.mins.size()}\n"
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, x):
        device = self.mins.device
        new_x = torch.cat((self.X, x.cpu()), dim=0)
        self.__init__(new_x)
        self.to_device(device)

    def to_device(self, device):
        self.device = device
        self.mins = self.mins.to(device)
        self.maxs = self.maxs.to(device)

class DebugNormalizer(Normalizer):
    """
    identity function
    """

    def normalize(self, x, *args, **kwargs):
        return x

    def unnormalize(self, x, *args, **kwargs):
        return x

class LimitsNormalizer(Normalizer):
    """
    maps [ xmin, xmax ] to [ -1, 1 ]
    """

    def normalize(self, x):
        x = (x - self.mins) / (self.maxs - self.mins)
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):

        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = torch.clip(x, -1, 1)
        x = (x + 1) / 2.0

        return x * (self.maxs - self.mins) + self.mins

class SafeLimitsNormalizer(LimitsNormalizer):
    """
    functions like LimitsNormalizer, but can handle data for which a dimension is constant
    """
    def __init__(self, *args, eps=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Find where min == max (constant dimensions)
        constant_mask = (self.mins == self.maxs)
        
        if constant_mask.any():
            num_constant = constant_mask.sum().item()
            print(
                f"""
                [ utils/normalization ] Found {num_constant} constant dimensions | """
                f"""adjusting with eps = {eps}"""
            )
            # Adjust all constant dimensions
            self.mins = torch.where(constant_mask, self.mins - eps, self.mins)
            self.maxs = torch.where(constant_mask, self.maxs + eps, self.maxs)

class GaussianNormalizer(Normalizer):
    """
    normalizes to zero mean and unit variance
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = self.X.mean(dim=0)[0]
        self.stds = self.X.std(dim=0)[0]
        self.z = 1

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size()}\n    """
            f"""means: {self.means}\n    """
            f"""stds: {self.z * self.stds}\n"""
        )

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means

    def to_device(self, device):
        self.device = device
        self.means = self.means.to(device)
        self.stds = self.stds.to(device)               