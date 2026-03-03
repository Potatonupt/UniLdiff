import sys

sys.path.insert(0, '/data2/czh/code/faithdiff/NAFNet')
from NAFNet.basicsr.models.archs.NAFNet_arch import NAFNet
import torch
import argparse
import os
import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import lpips
import math
import numpy as np


class NAFNet_Combine(NAFNet):
    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        weight = 1
        x = x * weight + inp[:, 3:6, :, :]

        return x[:, :, :H, :W]


class TestDataset(Dataset):
    def __init__(self, lq_dir, gt_dir):
        self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, '*.[pj]*[np]*[g]*')))
        self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.[pj]*[np]*[g]*')))
        print(len(self.lq_paths), len(self.gt_paths))
        assert len(self.lq_paths) == len(self.gt_paths)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, idx):
        lq = Image.open(self.lq_paths[idx]).convert("RGB")
        gt = Image.open(self.gt_paths[idx]).convert("RGB")
        lq = check_image_size(lq)
        lq = self.transform(lq)
        gt = self.transform(gt)
        return lq, gt, self.lq_paths[idx]


# 计算 PSNR
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))


# 反归一化函数
def denormalize(tensor):
    return tensor * 0.5 + 0.5


def check_image_size(x, padder_size=8):
    width, height = x.size

    if width % padder_size == 0 and height % padder_size == 0:
        return x
    else:
        # 向下取整为 padder_size 的倍数
        target_w = (width // padder_size) * padder_size
        target_h = (height // padder_size) * padder_size

        # resize 图像到新尺寸
        x_resized = x.resize((target_w, target_h), Image.LANCZOS)
        return x_resized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_in_path', type=str, default='LQ')
    parser.add_argument('--test_gt_path', type=str, default='RS')
    parser.add_argument('--model_path', type=str, default='/data2/czh/code/faithdiff/nafmodel/epoch_200.pth')
    parser.add_argument('--save_path', type=str, default='/data2/czh/code/faithdiff/save/nafnet/toled')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = NAFNet_Combine(img_channel=6, width=64,
                           enc_blk_nums=[2, 2, 4, 8],
                           middle_blk_num=12,
                           dec_blk_nums=[2, 2, 2, 2])
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 载入测试数据
    dataset = TestDataset(lq_dir=args.test_in_path, gt_dir=args.test_gt_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 评估指标
    criterion_lpips = lpips.LPIPS(net='vgg').to(device)
    total_psnr, total_ssim, total_lpips = 0, 0, 0

    os.makedirs(args.save_path, exist_ok=True)

    with torch.no_grad():
        for lq, gt, img_path in dataloader:
            lq, gt = lq.to(device), gt.to(device)

            # 扩展输入通道
            recon_stack = torch.cat((lq, gt), dim=1)
            pred = model(recon_stack)

            # 反归一化并保存
            save_image(denormalize(pred), os.path.join(args.save_path, os.path.basename(img_path[0])))
