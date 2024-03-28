import math
import torch

from dataclasses import dataclass, replace, field
from functools import partial
from typing import Callable, Optional, Union, Tuple, List
from torch import nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import Mlp, ConvMlp, DropPath, LayerNorm
from timm.layers import create_attn, get_act_layer, get_norm_layer, get_norm_act_layer, create_conv2d, create_pool2d
from timm.layers import trunc_normal_tf_, to_2tuple, extend_tuple, make_divisible, _assert
from timm.layers import RelPosMlp, RelPosBias, RelPosBiasTf
from timm.models._builder import build_model_with_cfg
from timm.models._features_fx import register_notrace_function
from timm.models._manipulate import named_apply, checkpoint_seq
from timm.models._pretrained import generate_default_cfgs
from timm.models._registry import register_model


__all__ = ['MaxxVitCfg', 'MaxxVitConvCfg', 'MaxxVitTransformerCfg', 'TVT']


@dataclass
class MaxxVitTransformerCfg:
    dim_head: int = 32
    head_first: bool = True  # head ordering in qkv channel dim
    expand_ratio: float = 4.0
    expand_first: bool = True
    shortcut_bias: bool = True
    attn_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.
    pool_type: str = 'avg2'
    rel_pos_type: str = 'bias_tf'
    rel_pos_dim: int = 512  # for relative position types w/ MLP
    partition_ratio: int = 32
    window_size: Optional[Tuple[int, int]] = None
    grid_size: Optional[Tuple[int, int]] = None
    no_block_attn: bool = False  # disable window block attention for maxvit (ie only grid)
    use_nchw_attn: bool = False  # for MaxViT variants (not used for CoAt), keep tensors in NCHW order
    init_values: Optional[float] = None
    act_layer: str = 'gelu'
    norm_layer: str = 'layernorm2d'
    norm_layer_cl: str = 'layernorm'
    norm_eps: float = 1e-6

    def __post_init__(self):
        if self.grid_size is not None:
            self.grid_size = to_2tuple(self.grid_size)
        if self.window_size is not None:
            self.window_size = to_2tuple(self.window_size)
            if self.grid_size is None:
                self.grid_size = self.window_size


@dataclass
class MaxxVitConvCfg:
    block_type: str = 'mbconv'
    expand_ratio: float = 4.0
    expand_output: bool = True  # calculate expansion channels from output (vs input chs)
    kernel_size: int = 3
    group_size: int = 1  # 1 == depthwise
    pre_norm_act: bool = False  # activation after pre-norm
    output_bias: bool = True  # bias for shortcut + final 1x1 projection conv
    stride_mode: str = 'dw'  # stride done via one of 'pool', '1x1', 'dw'
    pool_type: str = 'avg2'
    downsample_pool_type: str = 'avg2'
    padding: str = ''
    attn_early: bool = False  # apply attn between conv2 and norm2, instead of after norm2
    attn_layer: str = 'se'
    attn_act_layer: str = 'silu'
    attn_ratio: float = 0.25
    init_values: Optional[float] = 1e-6  # for ConvNeXt block, ignored by MBConv
    act_layer: str = 'gelu'
    norm_layer: str = ''
    norm_layer_cl: str = ''
    norm_eps: Optional[float] = None

    def __post_init__(self):
        # mbconv vs convnext blocks have different defaults, set in post_init to avoid explicit config args
        assert self.block_type in ('mbconv', 'convnext')
        use_mbconv = self.block_type == 'mbconv'
        if not self.norm_layer:
            self.norm_layer = 'batchnorm2d' if use_mbconv else 'layernorm2d'
        if not self.norm_layer_cl and not use_mbconv:
            self.norm_layer_cl = 'layernorm'
        if self.norm_eps is None:
            self.norm_eps = 1e-5 if use_mbconv else 1e-6
        self.downsample_pool_type = self.downsample_pool_type or self.pool_type


@dataclass
class MaxxVitCfg:
    embed_dim: Tuple[int, ...] = (96, 192, 384, 768)
    depths: Tuple[int, ...] = (2, 3, 5, 2)
    block_type: Tuple[Union[str, Tuple[str, ...]], ...] = ('C', 'C', 'T', 'T')
    stem_width: Union[int, Tuple[int, int]] = 64
    stem_bias: bool = False
    conv_cfg: MaxxVitConvCfg = field(default_factory=MaxxVitConvCfg)
    transformer_cfg: MaxxVitTransformerCfg = field(default_factory=MaxxVitTransformerCfg)
    head_hidden_size: int = None
    weight_init: str = 'vit_eff'


class AttentionCl(nn.Module):
    """ Channels-last multi-head attention (B, ..., C) """
    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            dim_head: int = 32,
            bias: bool = True,
            expand_first: bool = True,
            head_first: bool = True,
            rel_pos_cls: Callable = None,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first and dim_out > dim else dim
        assert dim_attn % dim_head == 0, 'attn dim should be divisible by head_dim'
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim_attn * 3, bias=bias)
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_attn, dim_out, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        if self.head_first:
            q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        else:
            q, k, v = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.dim_head).transpose(1, 3).unbind(2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rel_pos is not None:
            attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
        elif shared_rel_pos is not None:
            attn = attn + shared_rel_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma


class Downsample2d(nn.Module):
    """ A downsample pooling module supporting several maxpool and avgpool modes
    * 'max' - MaxPool2d w/ kernel_size 3, stride 2, padding 1
    * 'max2' - MaxPool2d w/ kernel_size = stride = 2
    * 'avg' - AvgPool2d w/ kernel_size 3, stride 2, padding 1
    * 'avg2' - AvgPool2d w/ kernel_size = stride = 2
    """
    def __init__(
            self,
            dim: int,
            dim_out: int,
            pool_type: str = 'avg2',
            padding: str = '',
            bias: bool = True,
    ):
        super().__init__()
        assert pool_type in ('max', 'max2', 'avg', 'avg2')
        if pool_type == 'max':
            self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=padding or 1)
        elif pool_type == 'max2':
            self.pool = create_pool2d('max', 2, padding=padding or 0)  # kernel_size == stride == 2
        elif pool_type == 'avg':
            self.pool = create_pool2d(
                'avg', kernel_size=3, stride=2, count_include_pad=False, padding=padding or 1)
        else:
            self.pool = create_pool2d('avg', 2, padding=padding or 0)

        if dim != dim_out:
            self.expand = nn.Conv2d(dim, dim_out, 1, bias=bias)
        else:
            self.expand = nn.Identity()

    def forward(self, x):
        x = self.pool(x)  # spatial downsample
        x = self.expand(x)  # expand chs
        return x


def _init_transformer(module, name, scheme=''):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # vit like
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if 'mlp' in name:
                    nn.init.normal_(module.bias, std=1e-6)
                else:
                    nn.init.zeros_(module.bias)


def _init_conv(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class MbConvBlock(nn.Module):
    """ Pre-Norm Conv Block - 1x1 - kxk - 1x1, w/ inverted bottleneck (expand)
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            drop_path: float = 0.
    ):
        super(MbConvBlock, self).__init__()
        norm_act_layer = partial(get_norm_act_layer(cfg.norm_layer, cfg.act_layer), eps=cfg.norm_eps)
        mid_chs = make_divisible((out_chs if cfg.expand_output else in_chs) * cfg.expand_ratio)
        groups = num_groups(cfg.group_size, mid_chs)

        if stride == 2:
            self.shortcut = Downsample2d(
                in_chs, out_chs, pool_type=cfg.pool_type, bias=cfg.output_bias, padding=cfg.padding)
        else:
            self.shortcut = nn.Identity()

        assert cfg.stride_mode in ('pool', '1x1', 'dw')
        stride_pool, stride_1, stride_2 = 1, 1, 1
        if cfg.stride_mode == 'pool':
            stride_pool, dilation_2 = stride, dilation[1]

        elif cfg.stride_mode == '1x1':
            stride_1, dilation_2 = stride, dilation[1]
        else:
            stride_2, dilation_2 = stride, dilation[0]

        self.pre_norm = norm_act_layer(in_chs, apply_act=cfg.pre_norm_act)
        if stride_pool > 1:
            self.down = Downsample2d(in_chs, in_chs, pool_type=cfg.downsample_pool_type, padding=cfg.padding)
        else:
            self.down = nn.Identity()
        self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=stride_1)
        self.norm1 = norm_act_layer(mid_chs)

        self.conv2_kxk = create_conv2d(
            mid_chs, mid_chs, cfg.kernel_size,
            stride=stride_2, dilation=dilation_2, groups=groups, padding=cfg.padding)

        attn_kwargs = {}
        if isinstance(cfg.attn_layer, str):
            if cfg.attn_layer == 'se' or cfg.attn_layer == 'eca':
                attn_kwargs['act_layer'] = cfg.attn_act_layer
                attn_kwargs['rd_channels'] = int(cfg.attn_ratio * (out_chs if cfg.expand_output else mid_chs))

        if cfg.attn_early:
            self.se_early = create_attn(cfg.attn_layer, mid_chs, **attn_kwargs)
            self.norm2 = norm_act_layer(mid_chs)
            self.se = None
        else:
            self.se_early = None
            self.norm2 = norm_act_layer(mid_chs)
            self.se = create_attn(cfg.attn_layer, mid_chs, **attn_kwargs)

        self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=cfg.output_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.pre_norm(x)
        x = self.down(x)

        x = self.conv1_1x1(x)
        x = self.norm1(x)

        x = self.conv2_kxk(x)
        if self.se_early is not None:
            x = self.se_early(x)
        x = self.norm2(x)
        if self.se is not None:
            x = self.se(x)

        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut
        return x


def window_partition(x, window_size: List[int]):
    B, H, W, C = x.shape
    _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    _assert(W % window_size[1] == 0, '')
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


@register_notrace_function
def window_reverse(windows, window_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def grid_partition(x, grid_size: List[int]):
    B, H, W, C = x.shape
    _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    _assert(W % grid_size[1] == 0, '')
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


@register_notrace_function
def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x


def get_rel_pos_cls(cfg: MaxxVitTransformerCfg, window_size):
    rel_pos_cls = None
    if cfg.rel_pos_type == 'mlp':
        rel_pos_cls = partial(RelPosMlp, window_size=window_size, hidden_dim=cfg.rel_pos_dim)
    elif cfg.rel_pos_type == 'bias':
        rel_pos_cls = partial(RelPosBias, window_size=window_size)
    elif cfg.rel_pos_type == 'bias_tf':
        rel_pos_cls = partial(RelPosBiasTf, window_size=window_size)
    return rel_pos_cls


class PartitionAttentionCl(nn.Module):
    """ Grid or Block partition + Attn + FFN. NxC 'channels last' tensor layout.
    """

    def __init__(
            self,
            dim: int,
            partition_type: str = 'block',
            cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer(cfg.norm_layer_cl), eps=cfg.norm_eps)  # NOTE this block is channels-last
        act_layer = get_act_layer(cfg.act_layer)

        self.partition_block = partition_type == 'block'
        self.partition_size = to_2tuple(cfg.window_size if self.partition_block else cfg.grid_size)
        rel_pos_cls = get_rel_pos_cls(cfg, self.partition_size)

        self.norm1 = norm_layer(dim)
        self.attn = AttentionCl(
            dim,
            dim,
            dim_head=cfg.dim_head,
            bias=cfg.attn_bias,
            head_first=cfg.head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * cfg.expand_ratio),
            act_layer=act_layer,
            drop=cfg.proj_drop)
        self.ls2 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition_attn(self, x):
        img_size = x.shape[1:3]
        if self.partition_block:
            partitioned = window_partition(x, self.partition_size)
        else:
            partitioned = grid_partition(x, self.partition_size)

        partitioned = self.attn(partitioned)

        if self.partition_block:
            x = window_reverse(partitioned, self.partition_size, img_size)
        else:
            x = grid_reverse(partitioned, self.partition_size, img_size)
        return x

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class Time_SA(nn.Module):
    """self-designed time self-attention module"""

    def __init__(self, batch_size, in_channels, drop_path, temporal_aggregation_size=3, cfg=MaxxVitTransformerCfg()):
        super().__init__()
        self.num_heads = in_channels // cfg.dim_head
        self.drop_path = drop_path
        self.batch_size = batch_size
        self.temporal_aggregation_size = temporal_aggregation_size
        self.dim_head = cfg.dim_head
        self.scale = cfg.dim_head ** -0.5
        self.attn_drop = nn.Dropout(cfg.attn_drop)
        self.proj = nn.Linear(in_channels, in_channels, bias=True)
        self.proj_drop = nn.Dropout(cfg.proj_drop)
        self.kvq = nn.Linear(in_channels, in_channels * 3, bias=True)
        norm_layer = partial(get_norm_layer(cfg.norm_layer_cl), eps=cfg.norm_eps)  # NOTE this block is channels-last
        act_layer = get_act_layer(cfg.act_layer)
        self.norm1 = norm_layer(in_channels)
        self.ls1 = LayerScale(in_channels, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_channels)
        self.mlp = Mlp(in_features=in_channels, hidden_features=int(in_channels * cfg.expand_ratio),
                       act_layer=act_layer, drop=cfg.proj_drop)
        self.ls2 = LayerScale(in_channels, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition_attn(self, x):
        B, H, W, C = x.shape
        kvq_input = x.reshape(-1, self.temporal_aggregation_size, H, W, C).permute(0, 2, 3, 1, 4).contiguous().view(-1, self.temporal_aggregation_size, C)
        k, v, q = self.kvq(kvq_input).view(-1, self.temporal_aggregation_size, 3, self.num_heads, self.dim_head).transpose(1, 3).unbind(2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).contiguous().view(-1, self.temporal_aggregation_size, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))\
            + x.reshape(-1, self.temporal_aggregation_size, H, W, C).permute(0, 2, 3, 1, 4).contiguous().view(-1, self.temporal_aggregation_size, C)
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = x.reshape(-1, H, W, self.temporal_aggregation_size, C).permute(0, 3, 1, 2, 4).contiguous().view(B, H, W, C)
        return x


class TVTBlock(nn.Module):
    """ MaxVit conv, window partition + FFN , grid partition + FFN
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            batch_size: int,
            temporal_aggregation_size: int = 3,
            stride: int = 1,
            conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
            use_time_SA=True,
    ):
        super().__init__()
        self.use_time_SA = use_time_SA
        self.nchw_attn = transformer_cfg.use_nchw_attn
        self.temporal_aggregation_size = temporal_aggregation_size
        conv_cls = MbConvBlock
        self.conv = conv_cls(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)

        attn_kwargs = dict(dim=dim_out, cfg=transformer_cfg, drop_path=drop_path)
        partition_layer = PartitionAttentionCl
        self.attn_block = partition_layer(**attn_kwargs)
        self.attn_grid = partition_layer(partition_type='grid', **attn_kwargs)
        if self.use_time_SA:
            self.attn_time = Time_SA(batch_size, dim_out, drop_path, temporal_aggregation_size, cfg=transformer_cfg)

    def init_weights(self, scheme=''):
        if self.attn_block is not None:
            named_apply(partial(_init_transformer, scheme=scheme), self.attn_block)
        named_apply(partial(_init_transformer, scheme=scheme), self.attn_grid)
        named_apply(partial(_init_conv, scheme=scheme), self.conv)

    def forward(self, x):
        # BCHW format
        x = self.conv(x)

        if not self.nchw_attn:
            x = x.permute(0, 2, 3, 1)  # to BHWC (channels-last)
        if self.attn_block is not None:
            x = self.attn_block(x)
        x = self.attn_grid(x)
        if self.use_time_SA:
            x = self.attn_time(x)
        if not self.nchw_attn:
            x = x.permute(0, 3, 1, 2)  # back to BCHW\
        return x


class TVTStage(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            batch_size: int,
            stride: int = 2,
            depth: int = 4,
            temporal_aggregation_size: int = 3,
            feat_size: Tuple[int, int] = (14, 14),
            block_types: Union[str, Tuple[str]] = 'C',
            transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            drop_path: Union[float, List[float]] = 0.,
            use_time_SA=True,
    ):
        super().__init__()
        self.depth = depth
        self.temporal_aggregation_size = temporal_aggregation_size
        block_types = extend_tuple(block_types, depth)
        blocks = []
        for i, t in enumerate(block_types):
            block_stride = stride if i == 0 else 1
            blocks += [TVTBlock(
                in_chs,
                out_chs,
                batch_size,
                temporal_aggregation_size=temporal_aggregation_size,
                stride=block_stride,
                conv_cfg=conv_cfg,
                transformer_cfg=transformer_cfg,
                drop_path=drop_path[i],
                use_time_SA=use_time_SA,
            )]

            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class Stem(nn.Module):

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            padding: str = '',
            bias: bool = False,
            act_layer: str = 'gelu',
            norm_layer: str = 'batchnorm2d',
            norm_eps: float = 1e-5,
    ):
        super().__init__()
        if not isinstance(out_chs, (list, tuple)):
            out_chs = to_2tuple(out_chs)

        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs[-1]
        self.stride = 2

        self.conv1 = create_conv2d(in_chs, out_chs[0], kernel_size, stride=2, padding=padding, bias=bias)
        self.norm1 = norm_act_layer(out_chs[0])
        self.conv2 = create_conv2d(out_chs[0], out_chs[1], kernel_size, stride=1, padding=padding, bias=bias)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x


def cfg_window_size(cfg: MaxxVitTransformerCfg, img_size: Tuple[int, int]):
    if cfg.window_size is not None:
        assert cfg.grid_size
        return cfg
    partition_size = img_size[0] // cfg.partition_ratio, img_size[1] // cfg.partition_ratio
    cfg = replace(cfg, window_size=partition_size, grid_size=partition_size)
    return cfg


def _overlay_kwargs(cfg: MaxxVitCfg, **kwargs):
    transformer_kwargs = {}
    conv_kwargs = {}
    base_kwargs = {}
    for k, v in kwargs.items():
        if k.startswith('transformer_'):
            transformer_kwargs[k.replace('transformer_', '')] = v
        elif k.startswith('conv_'):
            conv_kwargs[k.replace('conv_', '')] = v
        else:
            base_kwargs[k] = v
    cfg = replace(
        cfg,
        transformer_cfg=replace(cfg.transformer_cfg, **transformer_kwargs),
        conv_cfg=replace(cfg.conv_cfg, **conv_kwargs),
        **base_kwargs
    )
    return cfg


class TVT_main(nn.Module):
    """ MaxVit base model.Highly configurable for different block compositions, tensor layouts, pooling types.
    """

    def __init__(
            self,
            cfg: MaxxVitCfg,
            batch_size: int,
            img_size: Union[int, Tuple[int, int]] = 224,
            in_chans: int = 3,
            num_classes: int = 100,
            temporal_aggregation_size: int = 3,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            **kwargs,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        if kwargs:
            cfg = _overlay_kwargs(cfg, **kwargs)
        transformer_cfg = cfg_window_size(cfg.transformer_cfg, img_size)
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = cfg.embed_dim[-1]
        self.drop_rate = drop_rate
        self.temporal_aggregation_size = temporal_aggregation_size
        self.grad_checkpointing = False
        self.feature_info = []

        self.stem = Stem(
            in_chs=in_chans,
            out_chs=cfg.stem_width,
            padding=cfg.conv_cfg.padding,
            bias=cfg.stem_bias,
            act_layer=cfg.conv_cfg.act_layer,
            norm_layer=cfg.conv_cfg.norm_layer,
            norm_eps=cfg.conv_cfg.norm_eps,
        )
        stride = self.stem.stride
        self.feature_info += [dict(num_chs=self.stem.out_chs, reduction=2, module='stem')]
        feat_size = tuple([i // s for i, s in zip(img_size, to_2tuple(stride))])
        use_time_SA = [True, True, True, True]
        self.num_stages = len(cfg.embed_dim)
        assert len(cfg.depths) == self.num_stages
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        in_chs = self.stem.out_chs
        stages = []
        for i in range(self.num_stages):
            stage_stride = 2
            out_chs = cfg.embed_dim[i]
            feat_size = tuple([(r - 1) // stage_stride + 1 for r in feat_size])
            stages += [TVTStage(
                in_chs,
                out_chs,
                batch_size,
                temporal_aggregation_size=temporal_aggregation_size,
                depth=cfg.depths[i],
                block_types=cfg.block_type[i],
                conv_cfg=cfg.conv_cfg,
                transformer_cfg=transformer_cfg,
                feat_size=feat_size,
                drop_path=dpr[i],
                use_time_SA=use_time_SA[i],
            )]
            stride *= stage_stride
            in_chs = out_chs
            self.feature_info += [dict(num_chs=out_chs, reduction=stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        # Weight init (default PyTorch init works well for AdamW if scheme not set)
        assert cfg.weight_init in ('', 'normal', 'trunc_normal', 'xavier_normal', 'vit_eff')
        if cfg.weight_init:
            named_apply(partial(self._init_weights, scheme=cfg.weight_init), self)

    def _init_weights(self, module, name, scheme=''):
        if hasattr(module, 'init_weights'):
            try:
                module.init_weights(scheme=scheme)
            except TypeError:
                module.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["relative_position_bias_table", "rel_pos.mlp"])}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^stages\.(\d+)', None), (r'^norm', (99999,))]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def forward(self, x):
        x = self.stem(x)
        multiscale_feature = []
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            multiscale_feature.append(x)
        return multiscale_feature


def _rw_max_cfg(
        stride_mode='dw',
        pool_type='avg2',
        conv_output_bias=False,
        conv_attn_ratio=1 / 16,
        conv_norm_layer='',
        transformer_norm_layer='layernorm2d',
        transformer_norm_layer_cl='layernorm',
        window_size=None,
        dim_head=32,
        init_values=None,
        rel_pos_type='bias',
        rel_pos_dim=512,
):
    # 'RW' timm variant models were created and trained before seeing https://github.com/google-research/maxvit
    # Differences of initial timm models:
    # - mbconv expansion calculated from input instead of output chs
    # - mbconv shortcut and final 1x1 conv did not have a bias
    # - mbconv uses silu in timm, not gelu
    # - expansion in attention block done via output proj, not input proj
    return dict(
        conv_cfg=MaxxVitConvCfg(
            stride_mode=stride_mode,
            pool_type=pool_type,
            expand_output=False,
            output_bias=conv_output_bias,
            attn_ratio=conv_attn_ratio,
            act_layer='silu',
            norm_layer=conv_norm_layer,
        ),
        transformer_cfg=MaxxVitTransformerCfg(
            expand_first=False,
            pool_type=pool_type,
            dim_head=dim_head,
            window_size=window_size,
            init_values=init_values,
            norm_layer=transformer_norm_layer,
            norm_layer_cl=transformer_norm_layer_cl,
            rel_pos_type=rel_pos_type,
            rel_pos_dim=rel_pos_dim,
        ),
    )


def _next_cfg(
        stride_mode='dw',
        pool_type='avg2',
        conv_norm_layer='layernorm2d',
        conv_norm_layer_cl='layernorm',
        transformer_norm_layer='layernorm2d',
        transformer_norm_layer_cl='layernorm',
        window_size=None,
        no_block_attn=False,
        init_values=1e-6,
        rel_pos_type='mlp',  # MLP by default for maxxvit
        rel_pos_dim=512,
):
    # For experimental models with convnext instead of mbconv
    init_values = to_2tuple(init_values)
    return dict(
        conv_cfg=MaxxVitConvCfg(
            block_type='convnext',
            stride_mode=stride_mode,
            pool_type=pool_type,
            expand_output=False,
            init_values=init_values[0],
            norm_layer=conv_norm_layer,
            norm_layer_cl=conv_norm_layer_cl,
        ),
        transformer_cfg=MaxxVitTransformerCfg(
            expand_first=False,
            pool_type=pool_type,
            window_size=window_size,
            no_block_attn=no_block_attn,  # enabled for MaxxViT-V2
            init_values=init_values[1],
            norm_layer=transformer_norm_layer,
            norm_layer_cl=transformer_norm_layer_cl,
            rel_pos_type=rel_pos_type,
            rel_pos_dim=rel_pos_dim,
        ),
    )


def _tf_cfg():
    return dict(
        conv_cfg=MaxxVitConvCfg(
            norm_eps=1e-3,
            act_layer='gelu_tanh',
            padding='same',
        ),
        transformer_cfg=MaxxVitTransformerCfg(
            norm_eps=1e-5,
            act_layer='gelu_tanh',
            head_first=False,  # heads are interleaved (q_nh, q_hdim, k_nh, q_hdim, ....)
            rel_pos_type='bias_tf',
        ),
    )


model_cfgs = dict(
    maxvit_tiny_tf=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        stem_bias=True,
        head_hidden_size=512,
        **_tf_cfg())
)


def checkpoint_filter_fn(state_dict, model: nn.Module):
    model_state_dict = model.state_dict()
    out_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and v.ndim != model_state_dict[k].ndim and v.numel() == model_state_dict[k].numel():
            # adapt between conv2d / linear layers
            assert v.ndim in (2, 4)
            v = v.reshape(model_state_dict[k].shape)
        out_dict[k] = v
    return out_dict


def _create_TVT(variant, cfg_variant=None, pretrained=False, **kwargs):
    if cfg_variant is None:
        if variant in model_cfgs:
            cfg_variant = variant
        else:
            cfg_variant = '_'.join(variant.split('_')[:-1])
    return build_model_with_cfg(
        TVT_main, variant, pretrained,
        model_cfg=model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.95, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.conv1', 'classifier': 'head.fc',
        'fixed_input_size': True,
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'maxvit_tiny_tf_224.in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
})


@register_model
def TVT(pretrained=False, **kwargs):
    return _create_TVT('maxvit_tiny_tf_224', 'maxvit_tiny_tf', pretrained=pretrained, **kwargs)

