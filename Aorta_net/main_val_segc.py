import torch
torch.multiprocessing.set_sharing_strategy('file_system')##训练python脚本中import torch后，加上下面这句。
from load_datasets_transforms_seg import data_loader, data_transforms, infer_post_transforms,remove_regions
from monai.utils import first, set_determinism
from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR
from networks.mednext.MedNextV1 import MedNeXt as MedNeXt
from networks.SegResNet.network_backbone import SegResNet,SegResNetVAE
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import AsDiscrete,SaveImaged
from monai.metrics import DiceMetric

# from networks.DeMT_3D.network_backbone import DeMT
# from networks.PnPNet.unet import PnPNet
# from networks.PnPNet import vit_seg_configs as configs
# from networks.nnFormer.nnFormer_seg import nnFormer
# from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
# from monai.data.meta_tensor import MetaTensor

import os
import argparse
import yaml
import random
import numpy as np
import SimpleITK as sitk

def config():
    parser = argparse.ArgumentParser(description='3D UX-Net hyperparameters for medical image segmentation')
    ## Input data hyperparameters
    # parser.add_argument('--root', type=str, default='', required=True, help='Root folder of all your images and labels')
    parser.add_argument('--dataset', type=str, default='301', help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
    parser.add_argument('--train_list', type=str, default='./data/train.txt',help='Root folder')
    parser.add_argument('--val_list', type=str, default='./data/test.txt', help='Root folder')#因为要测试评分，所以使用validation代码
    parser.add_argument('--test_list', type=str, default='./data/test.txt',help='Root folder')
    parser.add_argument('--output', type=str, default='./output/MedNeXt', help='Output folder for both tensorboard and the best model')
    parser.add_argument('--image_save', type=str, default='/media/bit301/data/yml/data/p2_nii/test/MedNeXt', help='images')
    ## Input model & training hyperparameters
    #0128:large 无supervision
    #0203：large supervision
    #0210：large 添加卷积输出的supervision（opt-supervision）
    #0212：medium opt-supervision
    parser.add_argument('--network', type=str, default='MedNeXt',help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
    parser.add_argument('--trained_weights', default='./output/MedNeXt/0326/03_30_06.pth', help='Path of pretrained/fine-tuned weights')
    parser.add_argument('--supervision', type=bool,default=False, help='supervision')#True False
    parser.add_argument('--patch', type=int, default=(96,96,96), help='Batch size for subject input')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
    parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
    parser.add_argument('--crop_sample', type=int, default=1, help='Number of cropped sub-volumes for each subject')
    parser.add_argument('--sw_batch_size', type=int, default=2, help='Sliding window batch size for inference')
    parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

    ## Efficiency hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
    parser.add_argument('--cache_rate', type=float, default=0.2, help='Cache rate to cache your dataset into GPUs')#0.1
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers')

    args = parser.parse_args()
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
    set_determinism(seed=123)
    args=config() #the first running should excute this code
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # aa = np.random.rand(512,512,370)
    # aa = aa[np.newaxis, np.newaxis, :, :, :]
    # ab = torch.tensor(aa)  # .cuda()
    # ac = MetaTensor(ab).cuda()

    val_samples, out_classes = data_loader(args)
    val_files = [{"image": image_name, "label": label_name}
                 for image_name, label_name in
                 zip(val_samples['images'], val_samples['labels'])]
    # train_transforms, val_transforms = data_transforms(args)
    val_transforms = data_transforms(args)
    post_transforms_seg = infer_post_transforms("none",val_transforms, out_classes)  # test

    ## Inference Pytorch Data Loader and Caching
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    ## Load Networks
    ## Load Networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch=np.array(args.patch,dtype=int) #(96,96,48)
    patch = np.array(args.patch, dtype=int)  # (96,96,48)
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

    model.load_state_dict(torch.load(args.trained_weights))
    model.eval()
    post_label = AsDiscrete(to_onehot=out_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
    # dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)  # 去除背景项目
    dice_vals = list()
    with (torch.no_grad()):
        for i, val_data in enumerate(val_loader):#读取数据不对可能会导致数据加载报错
            torch.cuda.empty_cache()
            # if i<180:
            #     continue
            # images = val_data["image"].to(device)
            roi_size = patch  # roi_size=(96, 96, 96)
            val_inputs, val_labels = (val_data["image"].to(device), val_data["label"].to(device))#512x512x370
            # val_inputs = torch.Tensor(np.random.rand(1, 1, 512, 512, 370)).cuda()  # 2,192,192,4
            val_data['pred'] = sliding_window_inference(val_inputs, roi_size, 1, model,
                                                        overlap=args.overlap)#args.sw_batch_size

            val_label = [post_transforms_seg(i) for i in decollate_batch(val_data)]  # 调用SaveImaged保存为彩色（这里没有调用）
            # save(val_label[0]['pred'].numpy().squeeze())
            if args.supervision:
                label=val_label[0][0]['pred'].detach().cpu().numpy().squeeze().astype(np.int16)
            else:
                label=val_label[0]['pred'].detach().cpu().numpy().squeeze().astype(np.int16)
            label=label.transpose((2,1,0))
            # postlabels = remove_regions(label)  # .cuda()#仅仅保留最大连通区域
            label = sitk.GetImageFromArray(label)  # nibel读取格式为（z, x, y），sitk为（x, y，z）
            #save the
            path=val_files[i]["image"].split("0.nii")[0] #zx512x512
            path=path.split("external/")[1]
            path=os.path.join(args.image_save,path)
            if not os.path.isdir(path):
                os.makedirs(path)
            file_path = os.path.join(path,"2.nii.gz")
            sitk.WriteImage(label, file_path)
            i = i + 1
            if i % 10 == 0:
                print('numbers:', i)

            # aa=sitk.GetArrayFromImage(postlabels).astype(np.float32)
            # ab=torch.tensor(aa[np.newaxis, np.newaxis, :, :, :])
        #     val_outputs=MetaTensor(torch.tensor(label[np.newaxis,np.newaxis,:,:,:])).to(device)#label postlabels
        #     val_outputs_list= decollate_batch(val_outputs)
        #     if args.supervision:
        #         val_output_convert = [
        #             post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list[0]
        #         ]
        #     else:
        #         val_output_convert = [
        #             post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
        #         ]
        #     val_labels_list = decollate_batch(val_labels)
        #     val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
        #     dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        #     # dice = dice_metric.aggregate().item()
        #     # dice_vals.append(dice)
        #     dice = dice_metric.aggregate().cpu().detach().numpy()
        #
        # # mean_dice_val = np.mean(dice_vals)
        # # print("dice:",mean_dice_val)
        # sub_mean_dice = np.nanmean(dice, axis=0)  # 列平均
        # print("sub_mean_dice:", sub_mean_dice)
        #
        # total_mean_dice = np.mean(sub_mean_dice)  # 列平均
        # print("total_mean_dice:", total_mean_dice)

