import torch
from torch import nn
from torch import Tensor
class Amplitude(nn.Module):
    def __init__(self, inchannels, kernel_size=1,
                    stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Amplitude, self).__init__()
        self.fc = nn.Conv2d(inchannels, inchannels, kernel_size=1,
                    stride=1, padding=0, dilation=1, groups=1, bias=False)

    def forward(self,x):
        y = x * torch.cos(self.fc(x)) 
        return y   


class Amplitude_Pro(nn.Module):
    def __init__(self, inchannels):
        super(Amplitude_Pro, self).__init__()
        self.fc = DepthConvBlock(inchannels)

    def forward(self,x):
        y = x * torch.cos(self.fc(x)) + x 
        return y


class HighFilter(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        return self.block(x)


class HighFilterPro(nn.Module):
    def __init__(self, in_ch, depth_kernel=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, in_ch, depth_kernel=depth_kernel, stride=stride),
            ConvFFN(in_ch),
        )

    def forward(self, x):
        return self.block(x)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, depth_kernel=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, in_ch, depth_kernel=depth_kernel, stride=stride),
            ConvFFN(in_ch),
        )

    def forward(self, x):
        return self.block(x)

class ConvFFN(nn.Module):
    def __init__(self, in_ch1, in_ch2=None):
        super().__init__()

        if in_ch2 is None:
            in_ch2 = in_ch1 * 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch1, in_ch2, 1),
            nn.ReLU(),
            nn.Conv2d(in_ch2, in_ch1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        identity = x
        return identity + self.conv(x)


class DepthConv(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3=None, depth_kernel=3, stride=1):
        super().__init__()
        if in_ch3 is None:
            in_ch3 = in_ch2
            in_ch2 = in_ch1
        # dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch1, in_ch2, 1, stride=stride),
            nn.LeakyReLU(),
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_ch2, in_ch2, depth_kernel, padding=depth_kernel // 2, groups=in_ch2),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch2, in_ch3, 1),
            nn.LeakyReLU(),
        )
        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch1, in_ch3, 2, stride=2)
        elif in_ch1 != in_ch3:
            self.adaptor = nn.Conv2d(in_ch1, in_ch3, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity

class Round_with_grad(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input, None


def ste_round(x: Tensor) -> Tensor:
    """
    Rounding with non-zero gradients. Gradients are approximated by replacing
    the derivative by the identity function.

    Used in `"Lossy Image Compression with Compressive Autoencoders"
    <https://arxiv.org/abs/1703.00395>`_

    .. note::

        Implemented with the pytorch `detach()` reparametrization trick:

        `x_round = x_round - x.detach() + x`
    """

    return (torch.round(x) - x).detach() + x

if __name__ == '__main__':
    x = torch.rand(4, 192, 256, 256)