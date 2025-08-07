import torch
from mmengine.model import kaiming_init
from torch import nn
from torch.nn import BatchNorm2d
import torch.nn.functional as F

def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support zero-size tensor and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None




class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y).sigmoid()
        return x * y

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MDCL(nn.Module):
    def __init__(self, num_features=2048):
        super(MDCL, self).__init__()
        dropout0 = 0.1
        d_feature0 = 512
        d_feature1 = 256
        dim_in = num_features
        blob_out = "input"


        norm = "BN"
        self.db3 = Dilated_convolutionBlock(input_num=dim_in, num1=d_feature0, num2=d_feature1, dilation_rate=3,
                                    drop_out=dropout0)

        self.db6 = Dilated_convolutionBlock(input_num=dim_in + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                    dilation_rate=6, drop_out=dropout0)

        self.db12 = Dilated_convolutionBlock(input_num=dim_in + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=12, drop_out=dropout0)

        self.db18 = Dilated_convolutionBlock(input_num=dim_in + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=18, drop_out=dropout0)

        self.db24 = Dilated_convolutionBlock(input_num=dim_in + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=24, drop_out=dropout0)
        self.conv1 = Conv2d(
            in_channels=5 * d_feature1,
            out_channels=256,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            norm=get_norm(norm, 256),
        )
        self._init_weights()
        self.cbam = CBAM(channel=1280)
    def _init_weights(self):
        for layer in [self.conv1]:
            if layer is not None:
                kaiming_init(layer)

    def forward(self, feature):
        db3 = self.db3(feature)
        feature = torch.cat((feature, db3), dim=1)
        db6 = self.db6(feature)
        feature = torch.cat((feature, db6), dim=1)
        db12 = self.db12(feature)
        feature = torch.cat((feature, db12), dim=1)
        db18 = self.db18(feature)
        feature = torch.cat((feature, db18), dim=1)
        db24 = self.db24(feature)
        feature = torch.cat((db3, db6, db12, db18, db24), dim=1)
        feature = self.cbam(feature)
        return self.conv1(feature)

class Dilated_convolutionBlock(nn.Module):
    """ ConvNet block for building MDCL. """
    def __init__(self, input_num, num1, num2, dilation_rate, drop_out):
        super(Dilated_convolutionBlock,self).__init__()
        self.drop_out = drop_out
        norm="BN"
        self.conv1 = Conv2d(
            in_channels=input_num,
            out_channels=num1,
            kernel_size=(1,1),
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            norm=get_norm(norm, num1),
        )
        self.conv2 = Conv2d(
            in_channels=num1,
            out_channels=num2,
            kernel_size=(3,3),
            stride=1,
            padding=1 * dilation_rate,
            dilation=dilation_rate,
            bias=False,
        )
        self._init_weights()

    def _init_weights(self):
        for layer in [self.conv1, self.conv2]:
            if layer is not None:
                kaiming_init(layer)
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        out = F.relu_(out)
        return F.dropout(out,p=self.drop_out)