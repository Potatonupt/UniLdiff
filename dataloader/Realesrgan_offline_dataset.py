import os
import glob

import math
import torch
import random
import numpy as np
from PIL import Image
from functools import partial
import torch.nn.functional as F
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
from PIL import Image
import json
from transformers import CLIPImageProcessor
from torch import nn
from torchvision import transforms
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from .realesrgan import RealESRGAN_degradation
import cv2
import random
from glob import glob
from collections import OrderedDict
import yaml
from PIL import Image


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # ignore_security_alert_wait_for_fix RCE

    return opt


def convert_image_to_fn(img_type, image, minsize=512, eps=0.02):
    width, height = image.size
    if min(width, height) < minsize:
        scale = minsize / min(width, height) + eps
        image = image.resize((math.ceil(width * scale), math.ceil(height * scale)))

    if image.mode != img_type:
        return image.convert(img_type)
    return image


def exists(x):
    return x is not None


class LocalImageDataset(data.Dataset):
    def __init__(self,
                 dehazing_file=None,
                 deraining_file=None,
                 denoising_file=None,
                 deblurring_file=None,
                 low_light_file=None,
                 yml_kernel=None,
                 image_size=512,
                 tokenizer=None,
                 tokenizer_2=None,
                 center_crop=False,
                 random_flip=True,
                 resize_bak=True,
                 convert_image_to="RGB",
                 t_drop_rate=0.05
                 ):
        super(LocalImageDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2

        self.resize_bak = resize_bak

        self.crop_size = image_size

        self.t_drop_rate = t_drop_rate

        self.data_types = []
        self.data_prob = []

        nature_paths = []
        nature_lr_paths = []
        nature_jsons = []

        face_paths = []
        lq_face_paths = []
        face_jsons = []

        # Initialize lists to store image paths for each type
        dehazing_paths = []
        deraining_paths = []
        denoising_paths = []
        deblurring_paths = []
        low_light_paths = []

        dehazing_lr_paths = []
        deraining_lr_paths = []
        denoising_lr_paths = []
        deblurring_lr_paths = []
        low_light_lr_paths = []

        dehazing_jsons = []
        deraining_jsons = []
        denoising_jsons = []
        deblurring_jsons = []
        low_light_jsons = []
        extensions = ['*.png', '*.bmp', '*.jpg', '*.jpeg']
        if denoising_file is not None:
            for denoising_path_idx in denoising_file[0]:
                # denoising_path_list = sorted(glob(os.path.join(denoising_path_idx, '**', '*.png'), recursive=True))
                denoising_path_list = sorted(
                    sum([glob(os.path.join(denoising_path_idx, '**', ext), recursive=True) for ext in extensions], [])
                )
                denoising_paths += denoising_path_list

            for lq_denoising_path_idx in denoising_file[1]:
                # lq_img_path_list = sorted(glob(os.path.join(lq_denoising_path_idx, '**', '*.png'), recursive=True))
                lq_img_path_list = sorted(
                    sum([glob(os.path.join(lq_denoising_path_idx, '**', ext), recursive=True) for ext in extensions],
                        [])
                )
                denoising_lr_paths += lq_img_path_list

            for denoising_text_path_idx in denoising_file[2]:
                denoising_text_path_list = sorted(
                    glob(os.path.join(denoising_text_path_idx, '**', '*.json'), recursive=True))
                denoising_jsons += denoising_text_path_list

            # 如果 denoising 数据存在，添加到 data_types 中
            if len(denoising_paths) > 0:
                self.data_types.append('denoising')
                self.data_prob.append(0)  # 先给一个占位，之后调整概率

        # 处理 dehazing 数据
        if dehazing_file is not None:
            for dehazing_path_idx in dehazing_file[0]:
                dehazing_path_list = sorted(
                    sum([glob(os.path.join(dehazing_path_idx, '**', ext), recursive=True) for ext in extensions], [])
                )
                dehazing_paths += dehazing_path_list

            for lq_dehazing_path_idx in dehazing_file[1]:
                lq_img_path_list = sorted(
                    sum([glob(os.path.join(lq_dehazing_path_idx, '**', ext), recursive=True) for ext in extensions], [])
                )
                dehazing_lr_paths += lq_img_path_list

            for dehazing_text_path_idx in dehazing_file[2]:
                dehazing_text_path_list = sorted(
                    glob(os.path.join(dehazing_text_path_idx, '**', '*.json'), recursive=True))
                dehazing_jsons += dehazing_text_path_list

            if len(dehazing_paths) > 0:
                self.data_types.append('dehazing')
                self.data_prob.append(0)

        # 处理 deraining 数据
        if deraining_file is not None:
            for deraining_path_idx in deraining_file[0]:
                deraining_path_list = sorted(
                    sum([glob(os.path.join(deraining_path_idx, '**', ext), recursive=True) for ext in extensions], [])
                )
                deraining_paths += deraining_path_list

            for lq_deraining_path_idx in deraining_file[1]:
                lq_img_path_list = sorted(
                    sum([glob(os.path.join(lq_deraining_path_idx, '**', ext), recursive=True) for ext in extensions],
                        [])
                )
                deraining_lr_paths += lq_img_path_list

            for deraining_text_path_idx in deraining_file[2]:
                deraining_text_path_list = sorted(
                    glob(os.path.join(deraining_text_path_idx, '**', '*.json'), recursive=True))
                deraining_jsons += deraining_text_path_list

            if len(deraining_paths) > 0:
                self.data_types.append('deraining')
                self.data_prob.append(0)

        # 处理 deblurring 数据
        if deblurring_file is not None:
            for deblurring_path_idx in deblurring_file[0]:
                deblurring_path_list = sorted(
                    sum([glob(os.path.join(deblurring_path_idx, '**', ext), recursive=True) for ext in extensions], [])
                )
                deblurring_paths += deblurring_path_list

            for lq_deblurring_path_idx in deblurring_file[1]:
                lq_img_path_list = sorted(
                    sum([glob(os.path.join(lq_deblurring_path_idx, '**', ext), recursive=True) for ext in extensions],
                        [])
                )
                deblurring_lr_paths += lq_img_path_list

            for deblurring_text_path_idx in deblurring_file[2]:
                deblurring_text_path_list = sorted(
                    glob(os.path.join(deblurring_text_path_idx, '**', '*.json'), recursive=True))
                deblurring_jsons += deblurring_text_path_list

            if len(deblurring_paths) > 0:
                self.data_types.append('deblurring')
                self.data_prob.append(0)

        # 处理 low_light 数据
        if low_light_file is not None:
            for low_light_path_idx in low_light_file[0]:
                low_light_path_list = sorted(
                    sum([glob(os.path.join(low_light_path_idx, '**', ext), recursive=True) for ext in extensions], [])
                )
                low_light_paths += low_light_path_list

            for lq_low_light_path_idx in low_light_file[1]:
                lq_img_path_list = sorted(
                    sum([glob(os.path.join(lq_low_light_path_idx, '**', ext), recursive=True) for ext in extensions],
                        [])
                )
                low_light_lr_paths += lq_img_path_list

            for low_light_text_path_idx in low_light_file[2]:
                low_light_text_path_list = sorted(
                    glob(os.path.join(low_light_text_path_idx, '**', '*.json'), recursive=True))
                low_light_jsons += low_light_text_path_list

            if len(low_light_paths) > 0:
                self.data_types.append('low_light')
                self.data_prob.append(0)


        # Collect all the paths and JSON data for different tasks
        self.data_collection = {
            'dehazing': (np.array(dehazing_paths), np.array(dehazing_jsons), np.array(dehazing_lr_paths)),
            'deraining': (np.array(deraining_paths), np.array(deraining_jsons), np.array(deraining_lr_paths)),
            'denoising': (np.array(denoising_paths), np.array(denoising_jsons), np.array(denoising_lr_paths)),
            'deblurring': (np.array(deblurring_paths), np.array(deblurring_jsons), np.array(deblurring_lr_paths)),
            'low_light': (np.array(low_light_paths), np.array(low_light_jsons), np.array(low_light_lr_paths))
        }

        # Storing the lengths for each task
        self.data_lens = {
            'dehazing': len(dehazing_paths),
            'deraining': len(deraining_paths),
            'denoising': len(denoising_paths),
            'deblurring': len(deblurring_paths),
            'low_light': len(low_light_paths)
        }
        print("data_lens:")
        print(self.data_lens)

        self.data_lens_jsons = {
            'dehazing': len(dehazing_jsons),
            'deraining': len(deraining_jsons),
            'denoising': len(denoising_jsons),
            'deblurring': len(deblurring_jsons),
            'low_light': len(low_light_jsons)
        }
        print("data_lens_jsons:")
        print(self.data_lens_jsons)

        self.data_lens_lr = {
            'dehazing': len(dehazing_lr_paths),
            'deraining': len(deraining_lr_paths),
            'denoising': len(denoising_lr_paths),
            'deblurring': len(deblurring_lr_paths),
            'low_light': len(low_light_lr_paths)
        }
        print("data_lens_lr:")
        print(self.data_lens_lr)

        # 更新 data_prob，均分每个任务的概率
        num_valid_data_types = len(self.data_types)
        if num_valid_data_types > 0:
            equal_prob = 1.0 / num_valid_data_types
            self.data_prob = [equal_prob] * num_valid_data_types  # 平分概率

        # total_count = sum([self.data_lens[dt] for dt in self.data_types])
        # if total_count > 0:
        #     self.data_prob = [self.data_lens[dt] / total_count for dt in self.data_types]


        print(f"Data types: {self.data_types}")
        print(f"Data probabilities: {self.data_prob}")

        # self.data_collection = {'nature': (np.array(nature_paths), np.array(nature_jsons), np.array(nature_lr_paths)), 'face': (np.array(face_paths), np.array(face_jsons), np.array(lq_face_paths))}
        # self.data_lens = {'nature': len(nature_paths), 'face': len(face_paths)}
        # print(self.data_lens)
        # self.data_lens = {'nature': len(nature_jsons),  'face': len(face_jsons)}
        # print(self.data_lens)
        # self.data_lens = {'nature': len(nature_lr_paths), 'face': len(lq_face_paths)}
        # print(self.data_lens)
        def names_match(gt_name, lq_name, json_name):
            # 判断 LQ 和 GT 是否相等，或 GT 是 'no' + LQ
            lq_gt_match = (lq_name == gt_name) or (gt_name == 'no' + lq_name)
            # 判断 JSON 是否等于 GT 或 LQ
            json_match = (json_name == gt_name) or (json_name == lq_name)
            return lq_gt_match and json_match


        def check_correspondence(data_collection):
            for task, (gt_paths, json_paths, lq_paths) in data_collection.items():
                print(f"\nChecking task: {task}")

                # Check length consistency
                if not (len(gt_paths) == len(json_paths) == len(lq_paths)):
                    print(f"❌ Length mismatch: GT={len(gt_paths)}, JSON={len(json_paths)}, LQ={len(lq_paths)}")
                    continue

                # Check filename correspondence
                mismatch_found = False
                for i in range(len(gt_paths)):
                    gt_name = os.path.basename(gt_paths[i]).split('.')[0]
                    json_name = os.path.basename(json_paths[i]).split('.')[0]
                    lq_name = os.path.basename(lq_paths[i]).split('.')[0]

                    if not names_match(gt_name, lq_name, json_name):
                        print(f"❌ Mismatch at index {i}: GT={gt_name}, JSON={json_name}, LQ={lq_name}")
                        mismatch_found = True

                if not mismatch_found:
                    print("✅ All entries match in length and filenames.")

        # # 调用检查函数
        check_correspondence(self.data_collection)

    def __getitem__(self, index):

        data_type = random.choices(self.data_types, self.data_prob)[0]
        index = np.random.randint(self.data_lens[data_type])

        # load image
        img_path = self.data_collection[data_type][0][index]
        json_path = self.data_collection[data_type][1][index]
        lq_img_path = self.data_collection[data_type][2][index]
        gt_path = img_path
        data = json.load(open(json_path))
        init_text = data["caption"]
        words = init_text.split()
        words = words[3:]
        words[0] = words[0].capitalize()
        text = ' '.join(words)
        text = text.split('. ')
        text = '. '.join(text[:2]) + '.'

        image = Image.open(img_path).convert('RGB')
        lq_image = Image.open(lq_img_path).convert('RGB')

        w, h = lq_image.size
        pil_img = np.array(image)
        pil_lr_img = np.array(lq_image)
        pil_img, pil_lr_img = augment([pil_img, pil_lr_img], hflip=True, rotation=False)

        crop_pad_size = self.crop_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            pil_lr_img = cv2.copyMakeBorder(pil_lr_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            pil_img = cv2.copyMakeBorder(pil_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        # crop
        if pil_lr_img.shape[0] > crop_pad_size or pil_lr_img.shape[1] > crop_pad_size:
            h, w = pil_lr_img.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)

            pil_lr_img = pil_lr_img[top: top + crop_pad_size, left: left + crop_pad_size, ...]
            pil_img = pil_img[top: top + crop_pad_size, left: left + crop_pad_size, ...]

        else:
            top = 0
            left = 0

        lq_image = Image.fromarray(pil_lr_img)

        # Remove the 4x resizing for low-quality image (LR)
        image = Image.fromarray(pil_img)

        # Do not scale the coordinates as no super-resolved images are involved
        original_size = torch.tensor([h, w])
        crop_coords_top_left = torch.tensor([top, left])

        GT_image_t = np.asarray(image) / 255.
        LR_image_t = np.asarray(lq_image) / 255.

        GT_image_t, LR_image_t = img2tensor([GT_image_t, LR_image_t], bgr2rgb=False, float32=True)
        LR_image_t = LR_image_t * 2.0 - 1.0
        GT_image_t = GT_image_t * 2.0 - 1.0

        rand_num = random.random()
        if rand_num < self.t_drop_rate:
            text = ""

        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        null_text_input_ids = self.tokenizer(
            '',
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        null_text_input_ids_2 = self.tokenizer_2(
            '',
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return {
            'lq_image': LR_image_t,
            "image": GT_image_t,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([crop_pad_size, crop_pad_size]),
            'gt_path': gt_path,
            'null_text_input_ids': null_text_input_ids,
            "null_text_input_ids_2": null_text_input_ids_2
            # "check_img": check_image
        }

    def __len__(self):
        total_length = 0
        for key, value in self.data_lens.items():
            total_length += value
        return total_length
