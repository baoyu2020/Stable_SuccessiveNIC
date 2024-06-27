import torch 
from torch import nn
import models.torch_msssim as torch_msssim
import math

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device=0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.msssim = torch_msssim.MS_SSIM(max_val=1., device_id=device)
        self.lmbda = lmbda

    def forward(self, output, target, args):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_main"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)  
        out["bpp_hyper"] = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)  
        out["bpp_loss"] = out["bpp_main"] + out["bpp_hyper"]
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["msssim_loss"] = self.msssim(output["x_hat"], target)
        
        if args.metric == 'msssim':
            out["loss"] = self.lmbda * (1.0 - out["msssim_loss"]) + out["bpp_loss"]
        elif args.metric == 'identity':            
            out["loss"] = self.lmbda * 255 ** 2 *( out["mse_loss"] + output["identity_loss"] )+ out["bpp_loss"]
        elif args.metric == 'mse':
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            raise KeyError('Flag is Not Exist')
        return out