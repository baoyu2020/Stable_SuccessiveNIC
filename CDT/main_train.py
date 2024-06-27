import random, sys, os, time

import torch
from torchvision import transforms

from runx.logx import logx

from compressai.datasets import ImageFolder

from config import parse_args
sys.path.append("..")


from models.model_all import ScaleHyperpriorsAutoEncoder, JointAutoregressiveHierarchicalPriors, ChannelWise

from utils import *
from models.loss import RateDistortionLoss

from models.inference import train_one_epoch, test_epoch, test_multi_epoch

def main(argv):
    args = parse_args(argv)
    # print(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    # choice for model archs
    model_archs = [ScaleHyperpriorsAutoEncoder, JointAutoregressiveHierarchicalPriors, ChannelWise]
    model_names = ['M1', 'M2', 'M3', 'M4']
    net_pointer = model_archs[model_names.index(args.model)]

    if args.quality >= 5:
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
            args.N = 192
            print(f'Joint model with Low channel')
        elif args.model == 'M3':
            args.M = 320
            args.N = 192
            print(f'ChannelWise model with Low channel')
        else:
            args.M = 192
            args.N = 192
            print(f'Cheng2020Anchor with Low channel')


    model_path = os.path.join(args.out_dir, args.metric, 'models/')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    out_dir = os.path.join(args.out_dir, args.metric, str(args.quality), 'runlog/')
    logx.initialize(logdir=out_dir, coolname=False, tensorboard=True, hparams=vars(args), eager_flush=True)
    

    train_transforms = transforms.Compose([transforms.ToTensor()])

    test_transforms = transforms.Compose([transforms.ToTensor()])
    if args.metric == 'mse' or args.metric == 'identity':
        lambdas = {'1': 0.0016, '2': 0.0032, '3': 0.0075, 
                    '4': 0.015, '5': 0.03,  '6': 0.045} 
    elif args.metric == 'msssim':
        lambdas = { '1': 2.40, '2': 4.58, '3': 8.73,
                   '4': 16.64, '5': 31.73, '6': 60.50}        
    else:
        raise ValueError(f'Invalid metric!')

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="kodak", transform=test_transforms)

    train_dataloader = DataLoaderX(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    ) 

    test_dataloader = DataLoaderX(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )


    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    # torch.cuda.set_device(args.cuda)
    # device = torch.device('cuda')
    net = net_pointer(args.N, args.M, args.nt, args)
    logx.msg(f"Training Configuration:{args}\t")
    logx.msg(f"Networks:{net}\t")

    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    criterion = RateDistortionLoss(lmbda=lambdas[str(args.quality)], device=device)

    last_epoch = 0
    if args.resume:  # load from previous checkpoint
        cur_checkpoint = os.path.join(model_path, str(args.name) +'-best-'+ str(args.metric) +'-'+ str(args.quality)+ '.pth.tar')
        logx.msg(f"Loading:{cur_checkpoint}")
        checkpoint = torch.load(cur_checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"]
        # net.load_state_dict(checkpoint["state_dict"])
        net.load_state_dict({k.replace('module.', ''):v for k,v in checkpoint["state_dict"].items()})
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

    if args.pretrain:        
        logx.msg(f"Loading the pretrained model")
        args.epochs = 20
        logx.msg(f"The epoch of pretrain:{args.epochs}")
        model_dict = torch.load(os.path.join(model_path, str(args.name)+'-best-'+str(args.metric)+'-'+str(args.quality+1 )+r'.pth.tar'))
        # net.load_state_dict(model_dict["state_dict"])
        net.load_state_dict({k.replace('module.', ''):v for k,v in model_dict["state_dict"].items()})

    # if args.cuda and torch.cuda.device_count() > 1 :
    #     net = CustomDataParallel(net)

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        cur_lr = adjust_learning_rate(optimizer, epoch, args.learning_rate)
        logx.msg(f"Learning rate: {cur_lr}")
        st_time = time.time()
        train_one_epoch(            
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args,)            
        if (args.SIC>=2)&(epoch == (args.epochs-1)):
            loss = test_multi_epoch(epoch, test_dataloader, net, criterion, args)            
        else:
            loss = test_epoch(epoch, test_dataloader, net, criterion, args)
        # scheduler.step(loss)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        logx.msg(f"epoch time: {time.time() - st_time}")

        # if is_best:  
        if args.save:
            model_dir = os.path.join(args.out_dir, args.metric, 'models/')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if is_best:
                logx.msg(f'best epoch: {epoch+1}')
                Pathname = os.path.join(model_dir,
                            str(args.name)+'-'+'best' + '-' + str(args.metric) + '-' + str(int(args.quality)) + r'.pth.tar')
            else:
                Pathname = os.path.join(model_dir,
                            str(args.name)+'-'+'checkpoint' + '-' + str(args.metric) + '-' + str(int(args.quality)) + r'.pth.tar')

            torch.save(
                {
                    "cur_lr": cur_lr,
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                }, Pathname
            )

if __name__ == "__main__":
    main(sys.argv[1:])

