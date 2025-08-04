import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Callable, List

import torchvision
import torchvision.models.vision_transformer as vit

# --- Utility Functions ---

def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Recursively replaces submodules selected by a predicate with the output of a function.
    """
    if predicate(root_module):
        return func(root_module)

    module_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in module_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        
        tgt_module = func(src_module)
        
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module

def replace_bn_with_gn(current_module):
    """
    Recursively iterates through all sub-modules and replaces BatchNorm2d layers
    with GroupNorm layers. This function modifies the module in-place.
    """
    child_modules = list(current_module.named_children())
    for name, child_module in child_modules:
        if isinstance(child_module, nn.BatchNorm2d):
            num_channels = child_module.num_features
            num_groups = 16
            if num_channels % num_groups != 0:
                power_of_2 = 2**(num_groups.bit_length() - 1)
                while power_of_2 > 0 and num_channels % power_of_2 != 0:
                    power_of_2 //= 2
                num_groups = max(1, power_of_2)
            gn_layer = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
            setattr(current_module, name, gn_layer)
        elif len(list(child_module.children())) > 0:
            replace_bn_with_gn(child_module)
    return current_module # Return the modified module
# ==========================================================================================


# --- Backbone and Feature Extractor Classes ---

class SpatialBackbone(nn.Module):
    """
    A wrapper class to extract spatial feature maps (B, C, H, W) from various 
    backbone architectures. For CLIP, it can return the standard flattened feature vector
    or a spatial feature map if configured to do so. This class strips away the 
    final pooling and classification layers for most models.
    """
    def __init__(self, model: nn.Module):
        super().__init__()

        if model.__class__.__name__ == 'ResNet':
            self.backbone = nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4,
            )
        else:
            # Added a special check for R3M's convnet
            if 'ConvNet' in model.__class__.__name__ and hasattr(model, 'layer4'):
                self.backbone = model
            else:
                raise ValueError(f"Unsupported model architecture: {model.__class__.__name__}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Returns a spatial feature map of shape (B, C, H, W)
        """
        return self.backbone(x)


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape,
        num_kp=None,
        temperature=1.,
        learnable_temperature=True,
        output_variance=False,
        noise_std=0.00,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
                )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.reshape(-1, self._num_kp * 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


class VisionEncoders(nn.Module):
    def __init__(
        self,
        vision_model: str = "dino",
        views: List[str] = ["rgb"],
        replace_norm: bool = False,
        spatial_softmax: bool = False,
        num_kp : int = 0, 
        noise_std: float = 0.0,
        frozen: bool = True,
    ):
        super().__init__()
        self.views = views
        self.vision_model_name = vision_model
        
        print(f"\n--- Initializing VisionEncoder ---")
        print(f"Model: {vision_model}, Views: {views}, Replace Norm: {replace_norm}, Spatial Softmax: {spatial_softmax}, Frozen: {frozen}")

        # 1. Load the base model
        if vision_model == "r3m":
            try:
                import r3m
                model = r3m.load_r3m("resnet18").module.convnet.to('cpu')
            except ImportError:
                raise ImportError("R3M not found. Please install it using `pip install r3m`")
        elif vision_model == "e2e":
            model = torchvision.models.resnet18(weights=None) # From scratch
        elif vision_model == "imn":
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif vision_model == "dino":
            model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', verbose=False)
        else:
            raise ValueError(f"Vision model '{vision_model}' not recognized.")
        
        # 2. Optionally replace normalization layers
        if replace_norm:
            model = replace_bn_with_gn(model)

        # 3. Create the feature extractor
        feature_extractor = SpatialBackbone(model)
        
        # 4. Determine the final architecture (encoder pipeline)
        if spatial_softmax:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feature_shape = feature_extractor(dummy).shape[1:]
            if num_kp <= 0: # Handle invalid num_kp values
                num_kp = None
            encoder_pipeline = nn.Sequential(
                feature_extractor,
                SpatialSoftmax(input_shape=list(feature_shape), num_kp = num_kp, noise_std = noise_std)
            )
        else: # Not CLIP, not spatial_softmax
            encoder_pipeline = nn.Sequential(
                feature_extractor,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1)
            )
        
        # 5. Create a separate, deep-copied encoder for each view
        self.encoders = nn.ModuleDict()
        for view in views:
            self.encoders[view] = copy.deepcopy(encoder_pipeline)
            if frozen:
                self.encoders[view].requires_grad_(False)
                self.encoders[view].eval()

    def forward(self, x: Tensor, view: str) -> Tensor:
        if view not in self.views:
            raise ValueError(f"View '{view}' not in configured views: {self.views}")
        
        return self.encoders[view](x)

