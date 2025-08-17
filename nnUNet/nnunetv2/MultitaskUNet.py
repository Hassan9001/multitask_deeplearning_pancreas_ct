from typing import Union, List, Tuple, Type, Optional
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He, init_last_bn_before_add_to_0


class ResidualEncoderUNetWithClassifier(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = True,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        deep_supervision: bool = False,
        num_subtypes: int = 3,
        clf_cfe_channels: Tuple[int, int] = (128, 256),
        clf_fc_dims: Tuple[int, int, int, int] = (256, 128, 64, 32),
        clf_dropout: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.2),
        clf_pool_mode: str = "adaptive",
    ):
        super().__init__()
        self.encoder = ResidualEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, return_skips=True, disable_default_stem=False,
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        dim = convert_conv_op_to_dim(conv_op)
        if dim == 2:
            BN, GAP, Conv = nn.BatchNorm2d, nn.AdaptiveAvgPool2d, nn.Conv2d
        elif dim == 3:
            BN, GAP, Conv = nn.BatchNorm3d, nn.AdaptiveAvgPool3d, nn.Conv3d
        else:
            raise NotImplementedError("Only 2D/3D supported")

        bottleneck_c = features_per_stage[-1]
        cfe_layers: List[nn.Module] = []
        c_in = bottleneck_c
        for c_out in clf_cfe_channels:
            cfe_layers += [Conv(c_in, c_out, kernel_size=2, stride=1, padding=0, bias=False), BN(c_out), nn.ReLU(inplace=True)]
            c_in = c_out
        self.cfe = nn.Sequential(*cfe_layers)

        self.clf_pool_mode = clf_pool_mode
        self.gap = GAP(output_size=1) if clf_pool_mode == "adaptive" else nn.Identity()
        self._lazy_linear0: Optional[nn.Linear] = None

        fc_blocks: List[nn.Module] = []
        in_dim_placeholder = clf_fc_dims[0]
        for i, out_dim in enumerate(clf_fc_dims):
            p = clf_dropout[i] if i < len(clf_dropout) else 0.3
            fc_blocks += [nn.Linear(in_dim_placeholder, out_dim, bias=True), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True), nn.Dropout(p=p)]
            in_dim_placeholder = out_dim
        self.fc_blocks = nn.Sequential(*fc_blocks)
        self.fc_out = nn.Linear(clf_fc_dims[-1], num_subtypes, bias=True)
        self.bn_out = nn.BatchNorm1d(num_subtypes)

    @staticmethod
    def _flatten_features(x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=1)

    def _ensure_lazy_linear0(self, feat: torch.Tensor):
        if self._lazy_linear0 is None:
            self._lazy_linear0 = nn.Linear(feat.shape[1], self.fc_blocks[0].in_features, bias=True).to(feat.device, dtype=feat.dtype)

    def forward(self, x):
        skips = self.encoder(x)
        seg = self.decoder(skips)
        cfe = self.cfe(skips[-1])

        if self.clf_pool_mode == "adaptive":
            pooled = self.gap(cfe)
            feat = torch.flatten(pooled, start_dim=1)
            logits = self.fc_blocks(feat)
            logits = self.fc_out(logits)
            logits = self.bn_out(logits)
        else:
            flat = self._flatten_features(cfe)
            self._ensure_lazy_linear0(flat)
            feat0 = self._lazy_linear0(flat)
            logits = self.fc_blocks(feat0)
            logits = self.fc_out(logits)
            logits = self.bn_out(logits)
        return seg, logits

    def compute_conv_feature_map_size(self, input_size):
        enc = self.encoder.compute_conv_feature_map_size(input_size)
        dec = self.decoder.compute_conv_feature_map_size(input_size)
        return enc + dec

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)

        