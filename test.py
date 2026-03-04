import json

import torch.cuda
import argparse
from FaithDiff.create_FaithDiff_model import FaithDiff_pipeline
from PIL import Image
from CKPT_PTH import LLAVA_MODEL_PATH, SDXL_PATH, FAITHDIFF_PATH, VAE_FP16_PATH, BSRNet_PATH
from utils.color_fix import wavelet_color_fix, adain_color_fix
from utils.image_process import check_image_size
from llava.llm_agent import LLavaAgent

import os
import numpy as np
import cv2

from FaithDiff.create_FaithDiff_model import create_bsrnet
from utils.image_process import image2tensor, tensor2image

if torch.cuda.device_count() >= 2:
    LLaVA_device = 'cuda:1'
    Diffusion_device = 'cuda:0'
elif torch.cuda.device_count() == 1:
    Diffusion_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

# hyparams here
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--upscale", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--min_size", type=int, default=1024)
parser.add_argument("--latent_tiled_overlap", type=float, default=0.5)
parser.add_argument("--latent_tiled_size", type=int, default=1024)
parser.add_argument("--guidance_scale", type=float, default=5)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--no_llava", action='store_true', default=True)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--vae_tiled_overlap", type=float, default=0.25)
parser.add_argument("--vae_tiled_size", type=int, default=1024)
parser.add_argument("--use_bsrnet", action='store_true', default=False)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
parser.add_argument("--color_fix", type=str, choices=['wavelet', 'adain', 'nofix'], default='nofix')
parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr')
parser.add_argument("--cpu_offload", action='store_true', default=False)
parser.add_argument("--use_fp8", action='store_true', default=False)
parser.add_argument("--type", type=str)
parser.add_argument('--json_dir', type=str, default='./json_test/', help='Directory to save JSON captions')
args = parser.parse_args()
print(args)
cpu_offload = args.cpu_offload
use_fp8 = args.use_fp8
use_llava = not args.no_llava
use_bsrnet = args.use_bsrnet

# load FaithDiff FP16
pipe = FaithDiff_pipeline(sdxl_path=SDXL_PATH, VAE_FP16_path=VAE_FP16_PATH,
                          FaithDiff_path="/data2/czh/code/faithdiff/train_FaithDiff_stage_2_offline/tmp/checkpoint-14000/FaithDiff.bin",
                          use_fp8=use_fp8)
pipe = pipe.to(Diffusion_device)

if use_bsrnet:
    bsrnet = create_bsrnet(BSRNet_PATH)
    bsrnet.to(LLaVA_device)
    bsrnet.eval()
else:
    bsrnet = None

if args.use_tile_vae:
    ### enable_vae_tiling
    pipe.set_encoder_tile_settings()
    pipe.enable_vae_tiling()

if cpu_offload:
    pipe.enable_model_cpu_offload()

# load LLaVA
if use_llava:
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=LLaVA_device, load_8bit=args.load_8bit_llava, load_4bit=False)
else:
    llava_agent = None

# # 定义五个路径集合
input_dirs = [
    "/data2/czh/data/test/rainy1/LQ",
    # "/data2/czh/data/test/low-light/LQ",
    # "/data2/czh/data/test/SOTS/LQ",
    # "/data2/czh/data/test/motion-blurry/LQ",
    # "/data2/czh/data/test/noisy50/LQ",
    # "/data2/czh/data/test/noisy25/LQ",
    # "/data2/czh/data/test/noisy15/LQ"
    # "/data2/czh/data/test/TOLED_test/TOLED_LQ"
]

json_dirs = [
    "./json_test/rain100l",
    # "./json_test/low-light",
    # "./json_test/SOTS",
    # "./json_test/motion-blurry",
    # "./json_test/noise",
    # "./json_test/noise",
    # "./json_test/noise"
    # "./json_test/poled"
]

save_dirs = [
    "./save/rebut3_rain/rain100l",
    # "./save/tsne/low-light",
    # "./save/tsne/SOTS",
    # "./save/tsne/motion-blurry",
    # "./save/tsne/noisy50",
    # "./save/tsne/noisy25",
    # "./save/tsne/noisy15"
    # "./save/toled"
]

# 定义五个路径集合
# input_dirs = [
#     # "/data/czh/data/test/CDD11_test/haze",
#     "/data/czh/data/test/unseen",
#     "/data/czh/data/test/CDD11_test/haze_snow",
#     "/data/czh/data/test/CDD11_test/low",
#     "/data/czh/data/test/CDD11_test/low_haze",
#     "/data/czh/data/test/CDD11_test/low_haze_rain",
#     "/data/czh/data/test/CDD11_test/low_haze_snow",
#     "/data/czh/data/test/CDD11_test/low_rain",
#     "/data/czh/data/test/CDD11_test/low_snow",
#     "/data/czh/data/test/CDD11_test/rain",
#     "/data/czh/data/test/CDD11_test/snow",
# ]
#
# json_dirs = [
#     "./json_test/snow",
#     "./json_test/CDD11_test",
#     "./json_test/CDD11_test",
#     "./json_test/CDD11_test",
#     "./json_test/CDD11_test",
#     "./json_test/CDD11_test",
#     "./json_test/CDD11_test",
#     "./json_test/CDD11_test",
#     "./json_test/CDD11_test",
#     "./json_test/CDD11_test",
#     "./json_test/CDD11_test",
#
# ]
#
# save_dirs = [
#     # "./save/cdd/haze",
#     "./save/unseen/snow",
#     "./save/cdd/haze_snow",
#     "./save/cdd/low",
#     "./save/cdd/low_haze",
#     "./save/cdd/low_haze_rain",
#     "./save/cdd/low_haze_snow",
#     "./save/cdd/low_rain",
#     "./save/cdd/low_snow",
#     "./save/cdd/rain",
#     "./save/cdd/snow",
# ]

# input_dirs = [
#     # "/data/czh/data/train/allinone/Rain100L/LQ",
#     # "/data/czh/data/test/rainy1/LQ",
#     "/data/czh/data/train/allinone/low-light/LQ",
#     "/data/czh/data/train/allinone/OTS/LQ",
#     "/data/czh/data/train/allinone/motion-blurry/LQ",
#     "/data/czh/data/train/allinone/BSDWED50/LQ"
# ]
#
# json_dirs = [
#     # "./json/rain100l",
#     # "./json_test/rain100l",
#     "./json/low-light",
#     "./json/OTS",
#     "./json/motion-blurry",
#     "./json/noise"
# ]
#
# save_dirs = [
#     # "./save/fornaf/rain100l",
#     # "./save/fornaf/rain100l_2",
#     "./save/del/low-light",
#     "./save/del/SOTS",
#     "./save/del/motion-blurry",
#     "./save/del/noisy50"
# ]

def count_parameters(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name} - Total: {total_params/1e6:.2f} M, Trainable: {trainable_params/1e6:.2f} M")
    return total_params, trainable_params

count_parameters(pipe.unet, "UNet")
count_parameters(pipe.vae, "VAE")
count_parameters(pipe.text_encoder, "TextEncoder")
# 如果有 text_encoder_2：
if hasattr(pipe, "text_encoder_2"):
    count_parameters(pipe.text_encoder_2, "TextEncoder 2")


valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
for input_dir, json_dir, save_dir in zip(input_dirs, json_dirs, save_dirs):
    print(f"Processing: {input_dir}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    exist_file = os.listdir(save_dir)
    with torch.no_grad():
        for file_name in sorted(os.listdir(input_dir)):
            img_name, ext = os.path.splitext(file_name)
            if ext.lower() not in valid_extensions:
                continue  # 忽略非图像文件（包括 .json、.DS_Store 等）

            image = Image.open(os.path.join(input_dir, file_name)).convert('RGB')

            # 构造文件路径
            json_path = os.path.join(json_dir, img_name + '.json')

            # 尝试加载原始文件
            if os.path.isfile(json_path):
                with open(json_path, 'r') as f:
                    json_file = json.load(f)
            else:
                alt_json_path = os.path.join(json_dir, 'no' + img_name + '.json')
                if os.path.isfile(alt_json_path):
                    with open(alt_json_path, 'r') as f:
                        json_file = json.load(f)
                else:
                    # 如果还是找不到文件，可以定义一个备用方案或报错
                    raise FileNotFoundError(f"Neither {json_path} nor {alt_json_path} found.")
            init_text = json_file["caption"]
            words = init_text.split()
            words = words[3:]
            words[0] = words[0].capitalize()
            text = ' '.join(words)
            text = text.split('. ')
            text = '. '.join(text[:2]) + '.'

            print(text)

            # step 2: Restoration
            input_image, width_init, height_init, width_now, height_now = check_image_size(image)
            prompt_init = text
            negative_prompt_init = ""
            generator = torch.Generator(device='cuda').manual_seed(args.seed)
            gen_image = pipe(lr_img=input_image, prompt=prompt_init, negative_prompt=negative_prompt_init,
                             num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                             generator=generator, start_point=args.start_point, height=height_now, width=width_now,
                             overlap=args.latent_tiled_overlap,
                             target_size=(args.latent_tiled_size, args.latent_tiled_size)).images[0]
            if img_name.startswith('rain-'):
                img_name = "no" + img_name
            path = os.path.join(save_dir, img_name + '.png')

            # if (width_now != width_init) or (height_now != height_init):
            #     cropped_image = gen_image.resize((width_init, height_init), Image.LANCZOS)

            cropped_image = gen_image

            if args.color_fix == 'nofix':
                out_image = cropped_image
            else:
                if args.color_fix == 'wavelet':
                    out_image = wavelet_color_fix(cropped_image, image)
                elif args.color_fix == 'adain':
                    out_image = adain_color_fix(cropped_image, image)
            out_image.save(path)
