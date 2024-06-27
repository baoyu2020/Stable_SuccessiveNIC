import math
import os, warnings
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.models import CompressionModel, MeanScaleHyperprior

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d, conv3x3, subpel_conv3x3
from compressai.models.utils import conv, deconv, update_registered_buffers

from models.layer import (
    Amplitude, 
    Amplitude_Pro, 
    HighFilter, 
    HighFilterPro,
    ste_round
    )

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(
        min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class ScaleHyperpriorsAutoEncoder(CompressionModel):
    """autoencoder with a ScaleHyperpriors from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018."""

    def __init__(self, N, M, nt, args, **kwargs):

        super().__init__(**kwargs)      
        self.entropy_bottleneck= EntropyBottleneck(N)

        self.nt = nt
        if nt =='GDN':
            # "GDN baseline "
            self.encoder = nn.Sequential(
                conv(3, N),
                GDN(N), 
                conv(N, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                conv(N, M),         
            )
            self.decoder = nn.Sequential(
                deconv(M, N),            
                GDN(N, inverse= True),
                deconv(N, N),
                GDN(N, inverse= True), 
                deconv(N, N),
                GDN(N, inverse= True),
                deconv(N, 3),
            )
        elif nt =='CDT':
            self.Amplitude1 = Amplitude(N)
            self.Amplitude2 = Amplitude(N)
            self.Amplitude3 = Amplitude(N) 
            self.encoder = nn.Sequential(
                conv(3, N),
                self.Amplitude1,
                HighFilter(N),
                conv(N, N),
                self.Amplitude2,
                HighFilter(N),
                conv(N, N),
                self.Amplitude3,
                HighFilter(N),
                conv(N, M),            
            )

            self.decoder = nn.Sequential(
                deconv(M, N),            
                self.Amplitude3,             
                deconv(N, N),
                self.Amplitude2, 
                deconv(N, N),
                self.Amplitude1,
                deconv(N, 3),
            )
        elif nt =='CDT_Pro':
            self.Amplitude1 = Amplitude_Pro(N)
            self.Amplitude2 = Amplitude_Pro(N)
            self.Amplitude3 = Amplitude_Pro(N) 
            self.encoder = nn.Sequential(
                conv(3, N),
                self.Amplitude1,
                HighFilterPro(N),
                conv(N, N),
                self.Amplitude2,
                HighFilterPro(N),
                conv(N, N),
                self.Amplitude3,
                HighFilterPro(N),
                conv(N, M),            
            )
            self.decoder = nn.Sequential(
                deconv(M, N),            
                self.Amplitude3,                
                deconv(N, N),
                self.Amplitude2,                 
                deconv(N, N),
                self.Amplitude1,
                deconv(N, 3),
            )
        else:
            raise KeyError('Flag is Not Exist')            

######################################### hyperencoder ###################

        self.hyperencoder = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.hyperdecoder = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)        

        self.mse = nn.MSELoss()

    def forward(self, x):
        y =  self.encoder(x)
        z = self.hyperencoder(torch.abs(y))
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        scales = self.hyperdecoder(z_hat)
        _, y_likelihoods = self.gaussian_conditional(y, scales)
        y_hat = ste_round(y)
        x_hat = self.decoder(y_hat)
        x_hat = x_hat.clamp(0., 1.)
        y_hat_2 = self.encoder(x_hat)
        y_hat_2 = ste_round(y_hat_2)
        x_non_round = self.decoder(y).clamp(0., 1.)
        identity_loss = self.mse(y_hat, y_hat_2)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods, "z": z_likelihoods},
            "y": y,
            "z": z,
            "x_non": x_non_round,
            "identity_loss": identity_loss,
        }

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["encoder.0.weight"].size(0)
        M = state_dict["encoder.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        super().update(force=force)


class JointAutoregressiveHierarchicalPriors(MeanScaleHyperprior):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, nt, args, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        if nt =='GDN':
            # "GDN baseline "
            self.encoder = nn.Sequential(
                conv(3, N),
                GDN(N), 
                conv(N, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                conv(N, M),         
            )
            self.decoder = nn.Sequential(
                deconv(M, N),            
                GDN(N, inverse= True),
                deconv(N, N),
                GDN(N, inverse= True), 
                deconv(N, N),
                GDN(N, inverse= True),
                deconv(N, 3),
            )
        elif nt =='CDT':
            self.Amplitude1 = Amplitude(N)
            self.Amplitude2 = Amplitude(N)
            self.Amplitude3 = Amplitude(N) 
            self.encoder = nn.Sequential(
                conv(3, N),
                self.Amplitude1,
                HighFilter(N),
                conv(N, N),
                self.Amplitude2,
                HighFilter(N),
                conv(N, N),
                self.Amplitude3,
                HighFilter(N),
                conv(N, M),            
            )

            self.decoder = nn.Sequential(
                deconv(M, N),            
                self.Amplitude3,             
                deconv(N, N),
                self.Amplitude2, 
                deconv(N, N),
                self.Amplitude1,
                deconv(N, 3),
            )
        elif nt =='CDT_Pro':
            self.Amplitude1 = Amplitude_Pro(N)
            self.Amplitude2 = Amplitude_Pro(N)
            self.Amplitude3 = Amplitude_Pro(N) 
            self.encoder = nn.Sequential(
                conv(3, N),
                self.Amplitude1,
                HighFilterPro(N),
                conv(N, N),
                self.Amplitude2,
                HighFilterPro(N),
                conv(N, N),
                self.Amplitude3,
                HighFilterPro(N),
                conv(N, M),            
            )
            self.decoder = nn.Sequential(
                deconv(M, N),            
                self.Amplitude3,                
                deconv(N, N),
                self.Amplitude2,                 
                deconv(N, N),
                self.Amplitude1,
                deconv(N, 3),
            )
        else:
            raise KeyError('Flag is Not Exist')  

        self.hyperencoder = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.hyperdecoder = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        
        self.mse = nn.MSELoss()

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y =  self.encoder(x)
        z = self.hyperencoder(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        params = self.hyperdecoder(z_hat)
        _, means_y = params.chunk(2, 1)
        # y_hat = ste_round(y-means_y) + means_y # add noise y_hat
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        y_hat = ste_round(y)
        x_hat = self.decoder(y_hat)
        y_hat_2 = self.encoder(x_hat)
        y_hat_2 = ste_round(y_hat_2)
        x_hat = x_hat.clamp(0., 1.)
        x_non_round = self.decoder(y).clamp(0., 1.)
        identity_loss = self.mse(y_hat, y_hat_2)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods, "z": z_likelihoods},
            "y": y,
            "z": z,
            "x_non": x_non_round,
            "identity_loss": identity_loss,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net
        
    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv

#Channel-Wise 
class ChannelWise(CompressionModel):
    
    def __init__(self, N, M, nt, args, **kwargs):
        super().__init__(**kwargs)      
        self.entropy_bottleneck= EntropyBottleneck(N)


        self.num_slices = 10
        self.max_support_slices = 5

        if nt =='GDN':
            # "GDN baseline "
            self.encoder = nn.Sequential(
                conv(3, N),
                GDN(N), 
                conv(N, N),
                GDN(N),
                conv(N, N),
                GDN(N),
                conv(N, M),         
            )
            self.decoder = nn.Sequential(
                deconv(M, N),            
                GDN(N, inverse= True),
                deconv(N, N),
                GDN(N, inverse= True), 
                deconv(N, N),
                GDN(N, inverse= True),
                deconv(N, 3),
            )
        elif nt =='CDT':
            self.Amplitude1 = Amplitude(N)
            self.Amplitude2 = Amplitude(N)
            self.Amplitude3 = Amplitude(N) 
            self.encoder = nn.Sequential(
                conv(3, N),
                self.Amplitude1,
                HighFilter(N),
                conv(N, N),
                self.Amplitude2,
                HighFilter(N),
                conv(N, N),
                self.Amplitude3,
                HighFilter(N),
                conv(N, M),            
            )

            self.decoder = nn.Sequential(
                deconv(M, N),            
                self.Amplitude3,             
                deconv(N, N),
                self.Amplitude2, 
                deconv(N, N),
                self.Amplitude1,
                deconv(N, 3),
            )
        elif nt =='CDT_Pro':
            self.Amplitude1 = Amplitude_Pro(N)
            self.Amplitude2 = Amplitude_Pro(N)
            self.Amplitude3 = Amplitude_Pro(N) 
            self.encoder = nn.Sequential(
                conv(3, N),
                self.Amplitude1,
                HighFilterPro(N),
                conv(N, N),
                self.Amplitude2,
                HighFilterPro(N),
                conv(N, N),
                self.Amplitude3,
                HighFilterPro(N),
                conv(N, M),            
            )
            self.decoder = nn.Sequential(
                deconv(M, N),            
                self.Amplitude3,                
                deconv(N, N),
                self.Amplitude2,                 
                deconv(N, N),
                self.Amplitude1,
                deconv(N, 3),
            )
        else:
            raise KeyError('Flag is Not Exist')  

        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )

        self.h_scale_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
            )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.mse = nn.MSELoss()

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


    def forward(self, x):
        y = self.encoder(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)  

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            if self.ste is not True:
                y_hat_slice = ste_round(y_slice - mu) + mu
            # y_hat_slice = ste_round(y_slice - mu) + mu
            else:
                y_hat_slice = ste_round(y_slice)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.decoder(y_hat)

        y_hat_2 = self.encoder(x_hat)
        y_hat_2 = ste_round(y_hat_2)

        x_non_round = self.decoder(y)  
        identity_loss = self.mse(y_hat, y_hat_2)
        # print(identity_loss.shape)
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods, "z": z_likelihoods},
            "y": y,
            "z": z,
            "x_non": x_non_round,
            "identity_loss": identity_loss,
            }
            

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        # net = cls(192, 320)
        net.load_state_dict(state_dict)
        return net


    def compress(self, x):
        self.update(scale_table=get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS))
        y = self.g_a(x)
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []
        torch.cuda.synchronize()
        y_time = time.time()
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []        

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        torch.cuda.synchronize()
        y_enc_latency = time.time()-y_time

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:], "y_enc_latency":y_enc_latency}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_hat_slices = []
        
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        y_string = strings[0][0]
        decoder = RansDecoder()
        decoder.set_stream(y_string)

        torch.cuda.synchronize()
        y_time = time.time()
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)        
        
        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        torch.cuda.synchronize()
        y_dec_latency = time.time() - y_time
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat, "y_dec_latency":y_dec_latency}

