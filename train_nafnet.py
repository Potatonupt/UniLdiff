import sys
import os

# 添加本地 NAFNet 的 basicsr 路径到 sys.path 最前面
nafnet_root = '/data/czh/code/faithdiff/NAFNet'
sys.path.insert(0, os.path.join(nafnet_root))  # 添加根目录
sys.path.insert(0, os.path.join(nafnet_root, 'basicsr'))  # 添加 basicsr 路径

# 现在从本地导入，而不是 site-packages 中的
from NAFNet.basicsr.models.archs.NAFNet_arch import NAFNet



import time, argparse, sys, os
import torch, math, random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from torchvision.utils import save_image, make_grid
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
from PIL import Image
from pytorch_msssim import ssim
import lpips

# 创建保存路径
model_save_dir = "/data/czh/code/faithdiff/nafmodel/"
image_save_dir = "/data/czh/code/faithdiff/nafresult/"
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(image_save_dir, exist_ok=True)


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
        # print('NAFNet weighted: 1')

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        # weight of SCM
        weight = 1
        x = x * weight + inp[:, 3:6, :, :]

        return x[:, :, :H, :W]


parser = argparse.ArgumentParser()
parser.add_argument('--eval_in_path_L', type=str, default='/data2/czh/datasets/Test/Deraindrop/RainDrop_Valid/')
parser.add_argument('--eval_gt_path_L', type=str, default='/data2/czh/datasets/Test/Deraindrop/RainDrop_Valid/')

parser.add_argument('--eval_in_path_realRainDrop', type=str,
                    default='/data2/czh/datasets/Test/Deraindrop/RainDrop_Valid/')
parser.add_argument('--eval_gt_path_realRainDrop', type=str,
                    default='/data2/czh/datasets/Test/Deraindrop/RainDrop_Valid/')

parser.add_argument('--model_path', type=str, default='/home/czh/WGWS-Net/stage2res/WGWSS22/')
parser.add_argument('--model_name', type=str, default='net_epoch_119.pth')
parser.add_argument('--save_path', type=str, default='/home/czh/WGWS-Net/results/')

# training setting
parser.add_argument('--flag', type=str, default='s1')
parser.add_argument('--base_channel', type=int, default=18)
parser.add_argument('--num_block', type=int, default=6)
args = parser.parse_args()


class FusionDataset(Dataset):
    def __init__(self, lq_dirs, output_dirs, gt_dirs, patch_size=128, train=True, exts=["png", "jpg", "jpeg", "bmp"]):
        """
        lq_dirs, output_dirs, gt_dirs: List of directories.
        Each list should have the same length (e.g., 5 or more).
        """
        assert len(lq_dirs) == len(output_dirs) == len(gt_dirs), "Mismatched number of directories."

        self.lq_paths = []
        self.output_paths = []
        self.gt_paths = []

        # 遍历每组路径，将所有扩展名符合的图像加入列表
        for lq_dir, out_dir, gt_dir in zip(lq_dirs, output_dirs, gt_dirs):
            for ext in exts:
                self.lq_paths.extend(sorted(glob.glob(os.path.join(lq_dir, f"*.{ext}"))))
                self.output_paths.extend(sorted(glob.glob(os.path.join(out_dir, f"*.{ext}"))))
                self.gt_paths.extend(sorted(glob.glob(os.path.join(gt_dir, f"*.{ext}"))))

        assert len(self.lq_paths) == len(self.output_paths) == len(self.gt_paths), \
            "Number of lq, output, and gt images must match."

        self.train = train
        self.patch_size = patch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, idx):
        lq = Image.open(self.lq_paths[idx]).convert("RGB")
        output = Image.open(self.output_paths[idx]).convert("RGB")
        gt = Image.open(self.gt_paths[idx]).convert("RGB")

        if self.train:
            w, h = lq.size
            if w >= self.patch_size and h >= self.patch_size:
                left = random.randint(0, w - self.patch_size)
                top = random.randint(0, h - self.patch_size)
                box = (left, top, left + self.patch_size, top + self.patch_size)
                lq = lq.crop(box)
                output = output.crop(box)
                gt = gt.crop(box)

        lq = self.transform(lq)
        output = self.transform(output)
        gt = self.transform(gt)

        return lq, output, gt


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = NAFNet_Combine(img_channel=6, width=64,
                           enc_blk_nums=[2, 2, 4, 8],
                           middle_blk_num=12,
                           dec_blk_nums=[2, 2, 2, 2])
    model.to(device)

    # 载入数据
    dataset = FusionDataset(
        lq_dirs=
        [
            "/data/czh/data/test/low-light/LQ",
            "/data/czh/data/test/rainy1/LQ",
            "/data/czh/data/test/motion-blurry/LQ",
            "/data/czh/data/test/noisy50/LQ",
            "/data/czh/data/test/SOTS/LQ"
        ],
        output_dirs=
        [
            '/data/czh/code/faithdiff/save/epoch40000/low-light',
            '/data/czh/code/faithdiff/save/epoch40000/rain100l',
            '/data/czh/code/faithdiff/save/epoch40000/motion-blurry',
            '/data/czh/code/faithdiff/save/epoch40000/noisy50',
            '/data/czh/code/faithdiff/save/epoch40000/SOTS'
        ],
        gt_dirs=
        [
            '/data/czh/data/test/low-light/GT',
            '/data/czh/data/test/rainy1/GT',
            '/data/czh/data/test/motion-blurry/GT',
            '/data/czh/data/test/noisy50/GT',
            '/data/czh/data/test/SOTS/GT'
        ]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 损失函数
    criterion_l1 = nn.L1Loss()
    criterion_ssim = lambda x, y: 1 - ssim(x, y, data_range=1)  # SSIM 损失
    criterion_lpips = lpips.LPIPS(net='vgg').to(device)  # LPIPS 损失

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

    # 训练参数
    num_epochs = 100000
    best_loss = float("inf")
    save_interval = 1000  # 每 100 个 batch 保存图像

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (lq, output_img, gt) in enumerate(dataloader):
            lq, output_img, gt = lq.to(device), output_img.to(device), gt.to(device)

            # 拼接输入
            recon_stack = torch.cat((lq, output_img), dim=1)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            pred = model(recon_stack)

            # 计算损失
            loss_l1 = criterion_l1(pred, gt)
            loss_ssim = criterion_ssim(pred, gt)
            loss_lpips = criterion_lpips(pred, gt).mean()

            total_loss = loss_l1 + 0.2 * loss_ssim + 0.5 * loss_lpips  # 加权求和

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 记录损失
            epoch_loss += total_loss.item()

            # 每 50 个 batch 打印损失
            if batch_idx % 50 == 49:
                print(f"Epoch [{epoch + 1}/{num_epochs}] | Batch [{batch_idx + 1}] | "
                      f"L1: {loss_l1.item():.4f} | SSIM: {loss_ssim.item():.4f} | LPIPS: {loss_lpips.item():.4f} | Total: {total_loss.item():.4f}")

            # 每 save_interval 个 batch 保存拼接图像
            if batch_idx % save_interval == 0:
                batch_save_dir = os.path.join(image_save_dir, f"epoch_{epoch + 1}")
                os.makedirs(batch_save_dir, exist_ok=True)

                # 拼接图像
                lq_grid = make_grid(lq.cpu(), nrow=lq.size(0), normalize=True, scale_each=True)
                output_grid = make_grid(output_img.cpu(), nrow=output_img.size(0), normalize=True, scale_each=True)
                pred_grid = make_grid(pred.cpu(), nrow=pred.size(0), normalize=True, scale_each=True)
                gt_grid = make_grid(gt.cpu(), nrow=gt.size(0), normalize=True, scale_each=True)

                combined_grid = torch.cat((lq_grid, output_grid, pred_grid, gt_grid), dim=1)

                # 保存拼接好的大图
                save_image(combined_grid, os.path.join(batch_save_dir, f"batch_{batch_idx}.png"))

                print(f"Saved batch {batch_idx} images in {batch_save_dir}")

        # 计算 epoch 平均损失
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(model_save_dir, "best_nafnet_model.pth"))
            print(f"Saved new best model with loss: {best_loss:.4f}")

        # 仅在 epoch 为 100 的倍数时保存模型
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"epoch_{epoch + 1}.pth"))
            print(f"Saved model at epoch {epoch + 1}")

    # 最终模型保存
    torch.save(model.state_dict(), os.path.join(model_save_dir, "final_nafnet_model.pth"))
    print("Training complete. Final model saved.")
