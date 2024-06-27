
import numpy as np
import torch

from runx.logx import logx

import csv

from utils import *

def mse_to_db(mse):
    return 10 * np.log10(1.0 /mse)

def test_epoch(epoch, test_dataloader, model, criterion, args):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            d = d.unsqueeze(dim=0) if len(d.size()) == 3 else d             
            
            out_net = model(d)
            out_criterion = criterion(out_net, d, args)
            # aux_loss.update(model.aux_loss())
            loss.update(out_criterion["loss"])
            # loss.update(out_criterion["origin_loss"])
            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr_loss.update(10 * np.log10(1.0 / out_criterion["mse_loss"].item()))            


    logx.msg(
        f"=========test kodak===================\n"
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.6f} |"
        f"\tMSE loss: {mse_loss.avg:.4f} |"
        f"\tPSNR loss: {psnr_loss.avg:.4f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
    )
    logx.add_scalar('Test/Loss', loss.avg, epoch)
    logx.add_scalar('Test/bpp', bpp_loss.avg, epoch)
    logx.add_scalar('Test/PSNR', psnr_loss.avg, epoch)

    return loss.avg



def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args):
    model.train()
    device = next(model.parameters()).device
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d, args)
        out_criterion["loss"].backward()
        # out_criterion["loss"].backward(torch.ones(out_criterion["loss"].shape).to("cuda:0"))
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10000 == 0:
            psnr_loss = 10 * np.log10(1.0 / out_criterion["mse_loss"].item())
            logx.msg(
                f"Train epoch {epoch+1}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.6f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.4f} |'
                f'\tPSNR loss: {psnr_loss:.4f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.4f} |'
                f"\tAux loss: {aux_loss.item():.3f}"
            )


def test_multi_epoch(epoch, test_dataloader, model, criterion, args):
    model.eval()
    device = next(model.parameters()).device

    psnr_loss = AverageMeter()

    with torch.no_grad():
        Recon_loss_list=[]
        Recon_mse_list = []
        Recon_bpp_list = []
        count = 0
        for d in test_dataloader:
            count += 1
            d = d.to(device)
            d = d.unsqueeze(dim=0) if len(d.size()) == 3 else d               

            temp = d
            Recon_loss_temp=[]
            Recon_mse_temp = []
            Recon_bpp_temp = []
            for i in range(0, args.SIC):                    
                out_net = model(temp)
                temp = out_net["x_hat"]
                out_criterion = criterion(out_net, d, args)
                Recon_loss_temp.append(out_criterion["loss"].cpu().numpy())
                Recon_bpp_temp.append(out_criterion["bpp_loss"].cpu().numpy())
                Recon_mse_temp.append(out_criterion["mse_loss"].cpu().numpy())

            if count == 1:
                Recon_loss_list = Recon_loss_temp
                Recon_mse_list = Recon_mse_temp
                Recon_bpp_list = Recon_bpp_temp
            elif count == 2:
                Recon_loss_list = np.stack((Recon_loss_list, Recon_loss_temp), axis=0)
                Recon_mse_list = np.stack((Recon_mse_list, Recon_mse_temp), axis=0)
                Recon_bpp_list = np.stack((Recon_bpp_list, Recon_bpp_temp), axis=0)
            else:
                Recon_loss_temp = np.array(Recon_loss_temp)
                Recon_mse_temp = np.array(Recon_mse_temp)
                Recon_bpp_temp = np.array(Recon_bpp_temp)

                Recon_loss_list = np.append(Recon_loss_list, Recon_loss_temp.reshape(1, args.SIC), axis=0)
                Recon_mse_list = np.append(Recon_mse_list, Recon_mse_temp.reshape(1, args.SIC), axis=0)
                Recon_bpp_list = np.append(Recon_bpp_list, Recon_bpp_temp.reshape(1, args.SIC), axis=0)
        # print(Recon_bpp_list)
        Recon_loss_list = np.mean(Recon_loss_list, axis=0)
        Recon_mse_list = np.mean(Recon_mse_list, axis=0)
        Recon_bpp_list = np.mean(Recon_bpp_list, axis=0)
        Recon_dB_list = [mse_to_db(mse) for mse in Recon_mse_list]
        # print(Recon_bpp_list)

    if args.SIC == 1:
        psnr_loss = 10 * np.log10(1.0 /Recon_mse_list[0])
        logx.msg(
            f"=========test kodak===================\n"
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {Recon_loss_list[0]:.6f} |"
            f"\tMSE loss: {Recon_mse_list[0]:.4f} |"
            f"\tPSNR loss: {psnr_loss:.4f} |"
            f"\tBpp loss: {Recon_bpp_list[0]:.4f} |"

        )
        logx.add_scalar('Test/Loss', Recon_loss_list[0], epoch)
        logx.add_scalar('Test/bpp', Recon_bpp_list[0], epoch)
        logx.add_scalar('Test/PSNR', psnr_loss, epoch)

    else:
        psnr_one_loss = 10 * np.log10(1.0 /Recon_mse_list[0])
        psnr_last_loss = 10 * np.log10(1.0 /Recon_mse_list[-1])
        psnr_10_loss = 10 * np.log10(1.0 /Recon_mse_list[9])
        logx.msg(
            f"=========test kodak inference one time ===================\n"
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {Recon_loss_list[0]:.6f} |"
            f"\tMSE loss: {Recon_mse_list[0]:.4f} |"
            f"\tPSNR loss: {psnr_one_loss:.4f} |"
            f"\tBpp loss: {Recon_bpp_list[0]:.4f} |\n"
            f"=========test kodak inference {args.SIC} time ===================\n"
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {Recon_loss_list[-1]:.6f} |"
            f"\tMSE loss: {Recon_mse_list[-1]:.4f} |"
            f"\tPSNR loss: {psnr_last_loss:.4f} |"
            f"\tBpp loss: {Recon_bpp_list[-1]:.4f} | \n"
            ###中间数据
            f"=========test kodak inference 10 time ===================\n"
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {Recon_loss_list[9]:.6f} |"
            f"\tMSE loss: {Recon_mse_list[9]:.4f} |"
            f"\tPSNR loss: {psnr_10_loss:.4f} |"
            f"\tBpp loss: {Recon_bpp_list[9]:.4f} |"
        )
        for item in range(0, args.SIC):
            logx.msg(f"=========test kodak inference {item + 1} time ===================\n"
            f'\tRecon_loss_list: {Recon_loss_list[item]:.6f} |'
            f'\tRecon_mse_list: {Recon_mse_list[item]:.4f} |'
            f'\tRecon_bpp_list: {Recon_bpp_list[item]:.4f} |')

        logx.add_scalar('Test/Loss', Recon_loss_list[0], epoch)
        logx.add_scalar('Test/bpp', Recon_bpp_list[0], epoch)
        logx.add_scalar('Test/PSNR', psnr_one_loss, epoch)


        model_path = os.path.join(args.out_dir, args.metric, 'models/')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        out_dir = os.path.join(args.out_dir, args.metric, str(args.quality), 'runlog/multi_RD.csv')
        with open(out_dir, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Loss", "bpp", "mse", "mse_dB"])
            for row in zip(Recon_loss_list, Recon_bpp_list, Recon_mse_list, Recon_dB_list):
                writer.writerow(row)


        return Recon_loss_list[0]