import os
import re

# 替换成你的图片所在文件夹路径
folder_path = '/data2/czh/data/test/TOLED_test/TOLED_LQ'

# 遍历并重命名
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.webp')):
        name, ext = os.path.splitext(filename)
        # if name.isdigit():
        #     new_name = f"{int(name):03d}{ext}"  # 补齐3位数字
        #     old_path = os.path.join(folder_path, filename)
        #     new_path = os.path.join(folder_path, new_name)
        #     os.rename(old_path, new_path)
        #     print(f"Renamed: {filename} -> {new_name}")

        match = re.match(r"^(\d+)", name)
        if match:
            num = int(match.group(1))
            new_name = f"{num:04d}{ext}"  # 仅保留3位数字
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")