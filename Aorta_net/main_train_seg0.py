#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR
from networks.mednext.MedNextV1 import MedNeXt as MedNeXt
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss,DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch,PersistentDataset

import torch
torch.multiprocessing.set_sharing_strategy('file_system')##训练python脚本中import torch后，加上下面这句。
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms_seg import data_loader, data_transforms,DeepSupervisionWrapper,remove_regions

import os
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import random
import datetime
import gc
import math
import matplotlib.pyplot as plt

def config():
    parser = argparse.ArgumentParser(description='Aorta_Net hyperparameters for medical image segmentation')
    ## Input data hyperparameters
    # parser.add_argument('--root', type=str, default='', required=True, help='Root folder of all your images and labels')
    parser.add_argument('--dataset', type=str, default='301', help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
    parser.add_argument('--train_list', type=str, default='./data/train.txt',help='Root folder')
    parser.add_argument('--val_list', type=str, default='./data/val.txt', help='Root folder')
    parser.add_argument('--test_list', type=str, default='./data/val0.txt',help='Root folder')
    parser.add_argument('--output', type=str, default='./output/MedNeXtl/0322/', help='Output folder for both tensorboard and the best model')
    parser.add_argument('--image_save', type=str, default='./output/MedNeXtl/0322/', help='images')

    ## Input model & training hyperparameters
    #seg0:3DUXNET/SwinUNETR/MedNeXtl(supervision) (96,96,96)
    parser.add_argument('--network', type=str, default='nSwinUNETR', help='Network models: {SwinUNETR, 3DUXNET}')
    parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
    parser.add_argument('--pretrain', default="True", help='Have pretrained weights or not')#False
    parser.add_argument('--pretrained_weights', default='', help='Path of pretrained weights')
    parser.add_argument('--supervision', type=bool,default=True, help='supervision')#True False
    parser.add_argument('--patch', type=int, default=(96,96,96), help='Batch size for subject input')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default='2', help='Batch size for subject input')
    parser.add_argument('--crop_sample', type=int, default=1, help='Number of cropped sub-volumes for each subject')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
    parser.add_argument('--max_iter', type=int, default=80000, help='Maximum iteration steps for training')
    parser.add_argument('--eval_step', type=int, default=500, help='Per steps to perform validation')#4000 1e8

    ## Efficiency hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
    parser.add_argument('--cache_rate', type=float, default=0.01, help='Cache rate to cache your dataset into GPUs')#0.1  缓存数据占总数的百分比!!
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers')

    args = parser.parse_args()
    # with open('./Yaml/3DX-Unet1.yaml', 'w') as f:
    #     yaml.dump(args.__dict__, f, indent=2)
    return args

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True / False


if __name__ == '__main__':
    # seed_everything(seed=1234)
    set_determinism(seed=1234)
    args=config() #the first running should excute this code
    # input1=torch.Tensor(np.random.rand(2,1,128,128,80))#2,192,192,4
    # input2=torch.Tensor(np.random.rand(2,1,128,128,80))#2,192,192,4
    # x,y,z=model(input1)
    # intra_loss0 = intra_loss(64, 192, 5, 1 / 2)
    # loss0=intra_loss0(y,z,input2.squeeze())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('Used GPU: {}'.format(args.gpu))
    train_samples, valid_samples, out_classes = data_loader(args)

    train_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in
        zip(train_samples['images'], train_samples['labels'])
    ]

    val_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in
        zip(valid_samples['images'], valid_samples['labels'])
    ]
    train_transforms, val_transforms = data_transforms(args)

    # Train Pytorch Data Loader and Caching
    print('Start caching datasets!')
    train_ds = CacheDataset(data=train_files, transform=train_transforms,
                            cache_rate=args.cache_rate, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    # Valid Pytorch Data Loader and Caching
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    # val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir="./data/val")
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)#args.num_workers

    ## Load Networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch=np.array(args.patch,dtype=int)
    if args.network =='3DUXNET':##(1,1,96,96,96)
        model = UXNET(
            in_chans=1,
            out_chans=out_classes,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        ).to(device)
    elif args.network == 'SwinUNETR':
        model = SwinUNETR(
            img_size=patch,
            in_channels=1,
            out_channels=out_classes,
            feature_size=48,
            use_checkpoint=False,
        ).to(device)
    elif args.network == "MedNeXt":  ##(160,160,96)
        model = MedNeXt(
            in_channels=1,
            n_channels=32,
            n_classes=out_classes,
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            kernel_size=5,
            deep_supervision=args.supervision,
            do_res=True,
            do_res_up_down=True,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            checkpoint_style='outside_block'
        ).to(device)
    print('Chosen Network Architecture: {}'.format(args.network))

    if args.pretrain == 'False':# True False
        print('Pretrained weight is found! Start to load weight from: {}'.format(args.pretrained_weights))
        modd = "03_24_15.pth"
        model.load_state_dict(torch.load(os.path.join(args.output, modd)))

    # #计算参数时候记得放在CPU上进行，否则会报错
    # from thop import profile
    # input = torch.randn(1,1, 96, 96, 64)#96, 96, 96
    # # input = torch.randn(1,1, 64, 64, 64)
    # flops, params = profile(model, inputs=(input,))
    # print('flops:{}'.format(flops))
    # print('params:{}'.format(params))

    ## Define Loss function and optimizer
    # class_weight=torch.tensor([1,1,1,1.25,1.5]).cuda()##include_background ？
    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True,weight=class_weight)#class_weight
    if args.supervision:
        # mutilscale = [1, 1 / 2, 1 / 4, 1 / 8, 1/16]
        # weights = np.array([1, 1 / 2, 1 / 4, 1 / 8, 0])
        weights = np.array([1, 1 / 4, 1 / 8, 1 / 16, 0])
        weights = torch.tensor(weights / weights.sum())
        loss=DiceCELoss(to_onehot_y=True, softmax=True)
        loss_function=DeepSupervisionWrapper(loss,weights)
    else:
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)  # 原文中使用的损失函数
    # loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True)
    print('Loss for training: {}'.format('DiceCELoss'))
    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=1e-5)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)
    print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1000)

    root_dir = os.path.join(args.output)
    if os.path.exists(root_dir) == False:
        os.makedirs(root_dir)

    t_dir = os.path.join(root_dir, 'tensorboard')
    if os.path.exists(t_dir) == False:
        os.makedirs(t_dir)
    writer = SummaryWriter(log_dir=t_dir)

    def validation(epoch_iterator_val):
        model.eval()
        weight = np.array([1, 1, 1, 1])#dice weight
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                torch.cuda.empty_cache()
                val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
                # val_outputs = model(val_inputs)
                sw_batch_size = 1
                val_outputs = sliding_window_inference(val_inputs, patch, sw_batch_size, model)#(96, 96, 96)
                # val_outputs = model_seg(val_inputs, val_feat[0], val_feat[1])
                # val_labels_list = decollate_batch(val_labels)
                # val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)#
                if args.supervision:
                    val_output_convert = [
                        post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list[0]
                    ]
                else:
                    val_output_convert = [
                        post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                    ]
                dice_metric(y_pred=val_output_convert, y=val_labels)
                # dice = dice_metric.aggregate().item()
                # dice_vals1.append(dice)
                dice = dice_metric.aggregate().cpu().detach().numpy()#*(1,1,1.2,1.5)
                # dice_vals2.append(dice*weight)
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, np.nanmean(dice))
                )
                # dice_metric.reset() #使用这句就需要使用append
                # torch.cuda.empty_cache()
        sub_mean_dice = np.nanmean(dice, axis=0)
        # sub_mean_dice = np.nanmean(dice)
        print("Categories Dice: {}".format(sub_mean_dice))
        mean_dice_val = np.mean(sub_mean_dice*weight)
        writer.add_scalar('Validation Segmentation Loss', mean_dice_val, global_step)
        return mean_dice_val

    def train(global_step, train_loader, dice_val_best, global_step_best):
        # model_feat.eval()
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):
            step += 1
            # img=np.squeeze(batch["image"])[0,:,:,24]
            # plt.imshow(img,cmap='gray') # ,vmin=0,vmax=255
            # plt.show()

            if args.supervision:
                x = batch["image"].to(device)
                labels = batch["labels"]
                y = [label_scaled.to(device) for label_scaled in labels]
            else:
                x, y = batch["image"].to(device), batch["label"].to(device)
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
            )
            if (
                global_step % eval_num == 0 and global_step != 0
            ) or global_step == max_iterations:
                epoch_iterator_val = tqdm(
                    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )
                dice_val = validation(epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    modd = str(datetime.datetime.now().strftime('%m_%d_%H')) + ".pth"
                    torch.save(model.state_dict(), os.path.join(root_dir, modd))
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                    # scheduler.step(dice_val)
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                    # scheduler.step(dice_val)
            writer.add_scalar('Training Segmentation Loss', loss.data, global_step)
            global_step += 1
        return global_step, dice_val_best, global_step_best

    max_iterations = args.max_iter
    print('Maximum Iterations for training: {}'.format(str(args.max_iter)))
    eval_num = args.eval_step
    post_label = AsDiscrete(to_onehot=out_classes)
    post_pred = AsDiscrete(argmax=True, threshold=0.5,to_onehot=out_classes)
    # dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    # dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False)
    dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)  # 去除背景项目
    # intra_loss0 = intra_loss(64, 192, out_classes, 1 / 2)
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    global_step=0

    # subtrain_num = 5 #4:roder+4:shuffle+1 total
    # sub_files_num=int(len(train_files)/subtrain_num)#每次加载的数据量
    # batch=args.batch_size
    # global_step_add=math.ceil(sub_files_num/batch)#每批加载的数据执行完迭代才返回while
    # sub_iterations=global_step_add*50 #可以可以相等，也可以乘以倍数 100 50
    # kk=0
    # global_step=kk*sub_iterations #start
    # flag=sub_iterations+1
    while global_step < max_iterations:
        # k=int(global_step/sub_iterations)
        # totail=global_step%sub_iterations
        # if totail>=0 and totail<global_step_add:#分批训练  第一次余数
        # # if flag > sub_iterations:  # 分批训练  第一次余数
        #     print("training patch:"+str(k))
        #     flag = 0
        #     if k>kk:
        #         del train_loader
        #         del train_ds
        #         gc.collect()  # 强制进行垃圾回收
        #     sub_train_files = train_files[0:sub_files_num]
        #     train_ds = CacheDataset(data=sub_train_files, transform=train_transforms,
        #                             cache_rate=1, num_workers=args.num_workers)
        #     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        #                               pin_memory=True)
        # flag=flag+1
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )







