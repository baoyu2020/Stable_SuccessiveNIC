import argparse
import os
import time
import glob
import math
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import ptflops
import seaborn as sns
import matplotlib.pyplot as plt



from torchvision import transforms
from ptflops import get_model_complexity_info


from PIL import Image
from runx.logx import logx

from models.model_all import (ScaleHyperpriorsAutoEncoder, 
                            JointAutoregressiveHierarchicalPriors, 
                            ChannelWise,  
)

def mse_to_db(mse): 
    return 10 * np.log10(1.0 /mse)

def load_test(args):
    test_image_paths = []
    num = 0 
    dirs = os.listdir(args.dataset) #列出所有的input路径下文件
    num_test = len(dirs)
    print(f'the number of codec image is: {num_test}')
    for dir in dirs: 
        path = os.path.join(args.dataset, dir)
        if os.path.isdir(path):
            test_image_paths += glob.glob(path + '/*.png')[0:num_test]
        if os.path.isfile(path):
            if num < num_test:
                test_image_paths.append(path)
                num += 1
            
    return test_image_paths

def get_model_complexity(args):

    model_archs = [ScaleHyperpriorsAutoEncoder, JointAutoregressiveHierarchicalPriors, ChannelWise]
    model_names = ['M1', 'M2', 'M3']
    net_pointer = model_archs[model_names.index(args.model)]
    M = N = 1
    if args.q >= 4:
            args.M = 320
            args.N = 192
            print(f'High channel')
    else:
        if args.model == 'M1':
            args.M = 192
            args.N = 128
            print(f'Hyper model with Low channel')
        elif args.model == 'M2':
            args.M = 192
            args.N = 192  # 128
            print(f'Joint model with Low channel')
        elif args.model == 'M3':
            args.M = 320
            args.N = 192
            print(f'ChannelWise with Low channel')
        else:
            args.M = 192
            args.N = 128

    net = net_pointer(args.N, args.M, args.nt)
    macs, params = get_model_complexity_info(net, (3, 256, 256), print_per_layer_stat=True)
    
    return macs, params

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def image2tensor(image):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = Image.open(image) 
    ori_image =  np.array(img)
    source_img = transforms.ToTensor()(img).unsqueeze(0).to(device) #1, 3, H, W
    b, h, w = source_img.size(0), source_img.size(2), source_img.size(3)
    p = 64  
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        source_img,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )  #1, 3, h, w
    num_pixels = b * h * w

    return x_padded, num_pixels, padding_left, padding_right, padding_top, padding_bottom



def inference_mutird(im_dirs, args):
    """inference the model"""
    ############################## Load Configuration Parameters ########################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'The codec device is:{device}')

    model_archs = [ScaleHyperpriorsAutoEncoder, JointAutoregressiveHierarchicalPriors, ChannelWise]
    model_names = ['M1', 'M2', 'M3']
    net_pointer = model_archs[model_names.index(args.model)]

    if args.q >= 5:
            args.M = 320
            args.N = 192
            print(f'High channel')
    else:
        if args.model == 'M1':
            args.M = 192
            args.N = 128
            print(f'Hyper model with Low channel')
        elif args.model == 'M2':
            args.M = 192
            args.N = 128 #128
            print(f'Joint model with Low channel')
        elif args.model == 'M3':
            args.M = 320
            args.N = 192
            print(f'ChannelWise with Low channel')
        else:
            args.M = 320
            args.N = 192

    net = net_pointer(args.N, args.M, args.nt, args)
    ############################## Load Model ########################
    print(net)
    model_dict = torch.load(os.path.join(args.model_dir, str(args.name) + '-checkpoint-' + str(args.metric) + '-' + str(int(args.q)) + r'.pth.tar'), map_location='cuda:0') 
    net.load_state_dict({k.replace('module.', ''):v for k,v in model_dict["state_dict"].items()})
    print(f'load success! the best epoch:{model_dict["epoch"]}') 
    net = net.eval().to(device)
    ############################## Compress Each Image ########################
    test_dir = os.path.join(args.out_dir, args.name, str(args.q), 'test_file/')
    if os.path.exists(test_dir) is False:
        os.makedirs(test_dir)

    logx.initialize(logdir=test_dir, coolname=False, tensorboard=False, hparams=vars(args), eager_flush=True)
    ## Read Image ##
    # infer_time = 0
    # enc_dec_time_start = time.time()
    with torch.no_grad():        
        Recon_mse_list = []
        Recon_bpp_list = []
        count = 0
        for im_dir in im_dirs:
            print('====> Encoding Image:', im_dir)
            img = Image.open(im_dir)
            ori_image =  np.array(img)
            source_img = transforms.ToTensor()(img).unsqueeze(0).to(device) #1, 3, H, W
            # d = source_img
            b, h, w = source_img.size(0), source_img.size(2), source_img.size(3)
            x_padded, num_pixels, padding_left, padding_right, padding_top, padding_bottom = image2tensor(im_dir)
            d = x_padded
            count += 1
            d = d.to(device)          
            temp = d
            Recon_mse_temp = []
            Recon_bpp_temp = []
            x_last = None
            x_mid = None
            x_one = None
            for i in range(0, args.SIC):                    
                out_net = net.forward(temp)                
                temp = out_net["x_hat"]
                if i == 0:
                    x_one = temp
                if i == 9:
                    x_mid = temp
                if i ==(args.SIC-1):
                    x_last = temp 
                mse_loss = torch.nn.functional.mse_loss(temp, d)
                bpp_main = (torch.log(out_net["likelihoods"]["y"]).sum() / (
                            -math.log(2) * num_pixels)).item()  # bpp_main: E(-logP(y_hat|z_hat))
                bpp_hyper = (torch.log(out_net["likelihoods"]["z"]).sum() / (
                            -math.log(2) * num_pixels)).item()  # bpp_hyper: E(-logP(z_hat))
                # bpp_hyper =0
                bpp_loss = bpp_main + bpp_hyper

                Recon_bpp_temp.append(bpp_loss)
                Recon_mse_temp.append(mse_loss.cpu().numpy())

            if count == 1:
                Recon_mse_list = Recon_mse_temp
                Recon_bpp_list = Recon_bpp_temp
            elif count == 2: 
                Recon_mse_list = np.stack((Recon_mse_list, Recon_mse_temp), axis=0)
                Recon_bpp_list = np.stack((Recon_bpp_list, Recon_bpp_temp), axis=0)
            else:
                Recon_mse_temp = np.array(Recon_mse_temp)
                Recon_bpp_temp = np.array(Recon_bpp_temp)
                Recon_mse_list = np.append(Recon_mse_list, Recon_mse_temp.reshape(1, args.SIC), axis=0)
                Recon_bpp_list = np.append(Recon_bpp_list, Recon_bpp_temp.reshape(1, args.SIC), axis=0)
            
            if args.save:
                x_one = x_one.clamp(0.0, 1.0)
                one_image = transforms.ToPILImage()(x_one.squeeze())
                one_dir = os.path.join(test_dir, im_dir.split('/')[-1].replace('.png','-onedec.png'))
                one_image.save(one_dir)

                x_mid = x_mid.clamp(0.0, 1.0)
                mid_image = transforms.ToPILImage()(x_mid.squeeze())
                mid_dir = os.path.join(test_dir, im_dir.split('/')[-1].replace('.png','-10dec.png'))
                mid_image.save(mid_dir)

                x_last = x_last.clamp(0.0, 1.0)
                last_image = transforms.ToPILImage()(x_last.squeeze())
                last_dir = os.path.join(test_dir, im_dir.split('/')[-1].replace('.png','-lastdec.png'))
                last_image.save(last_dir)

        # print(Recon_bpp_list)
        Recon_mse_list = np.mean(Recon_mse_list, axis=0)
        Recon_bpp_list = np.mean(Recon_bpp_list, axis=0)
        Recon_dB_list = [mse_to_db(mse) for mse in Recon_mse_list]
        # print(Recon_bpp_list)

    if args.SIC == 1:
        psnr_loss = 10 * np.log10(1.0 /Recon_mse_list[0])
        logx.msg(
            f"=========test kodak===================\n"
            f"\tMSE loss: {Recon_mse_list[0]:.4f} |"
            f"\tPSNR loss: {psnr_loss:.4f} |"
            f"\tBpp loss: {Recon_bpp_list[0]:.4f} |"
        )
    else:
        psnr_one_loss = mse_to_db(Recon_mse_list[0])
        psnr_last_loss = mse_to_db(Recon_mse_list[-1])
        psnr_10_loss = mse_to_db(Recon_mse_list[9])
        logx.msg(
            f"=========test kodak inference one time ===================\n"
            f"\tMSE loss: {Recon_mse_list[0]:.4f} |"
            f"\tPSNR loss: {psnr_one_loss:.4f} |"
            f"\tBpp loss: {Recon_bpp_list[0]:.4f} |\n"
            ###中间数据
            f"=========test kodak inference 10 time ===================\n"
            f"\tMSE loss: {Recon_mse_list[9]:.4f} |"
            f"\tPSNR loss: {psnr_10_loss:.4f} |"
            f"\tBpp loss: {Recon_bpp_list[9]:.4f} | \n"
            f"=========test kodak inference {args.SIC} time ===================\n"
            f"\tMSE loss: {Recon_mse_list[-1]:.4f} |"
            f"\tPSNR loss: {psnr_last_loss:.4f} |"
            f"\tBpp loss: {Recon_bpp_list[-1]:.4f} | \n"
        )
        for item in range(0, args.SIC):
            logx.msg(f"=========test kodak inference {item + 1} time ===================\n"
            f'\tRecon_mse_list: {Recon_mse_list[item]:.4f} |'
            f'\tRecon_bpp_list: {Recon_bpp_list[item]:.4f} |')


        out_dir = os.path.join(test_dir, 'multi_RD.csv')
        with open(out_dir, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["bpp", "mse", "mse_dB"])
            for row in zip(Recon_bpp_list, Recon_mse_list, Recon_dB_list):
                writer.writerow(row)

       


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--dataset', type= str, default='/private/dataset/NIC_Dataset/CLIC2020/kodak', help= 'test dataset')
    parser.add_argument('-d', '--dataset', type= str, default='/private/dataset/NIC_Dataset/valid2020pro/', help= 'test dataset')
    
    parser.add_argument('-name', type= str, default= 'ICLR18', help= 'method name' )
    parser.add_argument('--save', action='store_true', help= 'save image' )
    parser.add_argument('--patch_size', type= int,nargs= 2,default= (1152, 1152), help= 'the cropped size--default: %(default)s')
    parser.add_argument( '--metric', type= str, default= 'mse', help= 'optimized metric')
    parser.add_argument( '-q', type= int, default= 3, required= True,  help = 'lamda value'  )
    parser.add_argument( '--out_dir', type = str, default= './test_logdir/debug', help= 'path of test log')
    parser.add_argument( '--model_dir', type = str, default= './Log/ShareCos_HighFilter_STE/mse/models', help = 'path of trained models')
    parser.add_argument('--nt', default='GDN', type=str, required=True, help='flag of Nonlinear Transform')
    parser.add_argument('--model', type=str, default='M1', choices=['M1', 'M2', 'M3'])
    parser.add_argument('--SIC', type= int, default= 50,  help='The inference time (default: %(default)s)')
    args = parser.parse_args()

    test_images = load_test(args)
    inference_mutird(test_images, args)
    # args.q = 0
    # for i in (0, 5):
    #     args.q = i + 1 
    #     inference_rd(test_images, args)

    # macs, params = get_model_complexity(args)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
