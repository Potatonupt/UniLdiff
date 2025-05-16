import os
import shutil

# 路径设置
gt_dir = "/data/czh/data/test/hazy1/GT"         # GT 是 .png
lq_dir = "/data/czh/data/test/hazy1/LQ"         # LQ 是 .jpg
out_gt_dir = "/data/czh/data/test/SOTS/GT"      # 输出 GT（重命名为 LQ 的名字，但保持 .png）
out_lq_dir = "/data/czh/data/test/SOTS/LQ"      # 输出 LQ

# 创建输出目录
os.makedirs(out_gt_dir, exist_ok=True)
os.makedirs(out_lq_dir, exist_ok=True)

# 构建 GT 文件索引（去掉扩展名）
gt_files = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir) if f.endswith('.png')}

# 遍历 LQ 文件
pair_count = 0
for lq_file in os.listdir(lq_dir):
    if not lq_file.endswith('.jpg'):
        continue

    prefix = lq_file.split('_')[0]  # 例如 '0001_0.8.jpg' -> '0001'
    if prefix in gt_files:
        gt_file = gt_files[prefix]

        # 原始 GT 路径
        src_gt_path = os.path.join(gt_dir, gt_file)

        # 目标 GT 路径：保留 lq_file 的文件名但强制后缀为 .png
        lq_name_without_ext = os.path.splitext(lq_file)[0]
        dst_gt_filename = lq_name_without_ext + ".png"
        dst_gt_path = os.path.join(out_gt_dir, dst_gt_filename)

        # 拷贝 GT 图像并重命名
        shutil.copy(src_gt_path, dst_gt_path)

        # 同时复制 LQ 图像
        shutil.copy(
            os.path.join(lq_dir, lq_file),
            os.path.join(out_lq_dir, lq_file)
        )

        pair_count += 1

print(f"共成功配对并重命名了 {pair_count} 对图像。")

