import torch

B, C, H, W = 2, 320, 8, 8
x = torch.randn(B, C, H, W)

# 展开
x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

x_recon = x_flat.transpose(1, 2).view(B, C, H, W)

print(torch.allclose(x, x_recon))  # True 表示完全对应
