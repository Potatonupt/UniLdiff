from transformers import CLIPTokenizer

from dataloader.Realesrgan_offline_dataset import LocalImageDataset


dehazing_file_path = ["/data/czh/data/train/CDD11_train/clear",
                      "/data/czh/data/train/CDD11_train/clear",
                      "/data/czh/data/train/CDD11_train/clear",
                      "/data/czh/data/train/CDD11_train/clear",
                      "/data/czh/data/train/CDD11_train/clear",
                      "/data/czh/data/train/CDD11_train/clear",
                      "/data/czh/data/train/CDD11_train/clear",
                      "/data/czh/data/train/CDD11_train/clear",
                      "/data/czh/data/train/CDD11_train/clear",
                      "/data/czh/data/train/CDD11_train/clear",
                      "/data/czh/data/train/CDD11_train/clear",]

lq_dehazing_file_path = ['/data/czh/data/train/CDD11_train/haze',
                         "/data/czh/data/train/CDD11_train/haze_rain",
                         "/data/czh/data/train/CDD11_train/haze_snow",
                         "/data/czh/data/train/CDD11_train/low",
                         "/data/czh/data/train/CDD11_train/low_haze",
                         "/data/czh/data/train/CDD11_train/low_haze_rain",
                         "/data/czh/data/train/CDD11_train/low_haze_snow",
                         "/data/czh/data/train/CDD11_train/low_rain",
                         "/data/czh/data/train/CDD11_train/low_snow",
                         "/data/czh/data/train/CDD11_train/rain",
                         "/data/czh/data/train/CDD11_train/snow",]

dehazing_json_file_path = ['./json/CDD11_train',
                           "./json/CDD11_train",
                           "./json/CDD11_train",
                           "./json/CDD11_train",
                           "./json/CDD11_train",
                           "./json/CDD11_train",
                           "./json/CDD11_train",
                           "./json/CDD11_train",
                           "./json/CDD11_train",
                           "./json/CDD11_train",
                           "./json/CDD11_train",]

# dehazing_file_path = ['/data/czh/data/train/allinone/OTS/GT']
# lq_dehazing_file_path = ['/data/czh/data/train/allinone/OTS/LQ']
# dehazing_json_file_path = ['./json/OTS']
#
# deraining_file_path = ['/data/czh/data/train/allinone/Rain100L/GT', '/data/czh/data/train/allinone/rain1800/GT']
# lq_deraining_file_path = ['/data/czh/data/train/allinone/Rain100L/LQ', '/data/czh/data/train/allinone/rain1800/LQ']
# deraining_json_file_path = ['./json/rain100l', './json/rain1800']
#
# denoising_file_path = ['/data/czh/data/train/allinone/BSDWED15/GT', '/data/czh/data/train/allinone/BSDWED25/GT',
#                        '/data/czh/data/train/allinone/BSDWED50/GT']
# lq_denoising_file_path = ['/data/czh/data/train/allinone/BSDWED15/LQ', '/data/czh/data/train/allinone/BSDWED25/LQ',
#                           '/data/czh/data/train/allinone/BSDWED50/LQ']
# denoising_json_file_path = ['./json/noise', './json/noise', './json/noise']
#
# deblurring_file_path = ['/data/czh/data/train/allinone/motion-blurry/GT']
# lq_deblurring_file_path = ['/data/czh/data/train/allinone/motion-blurry/LQ']
# deblurring_json_file_path = ['./json/motion-blurry']
#
# low_light_file_path = ['/data/czh/data/train/allinone/low-light/GT/']
# lq_low_light_file_path = ['/data/czh/data/train/allinone/low-light/LQ/']
# low_light_json_file_path = ['./json/low-light']

tokenizer = CLIPTokenizer.from_pretrained("/data/czh/models/RealVisXL_V4.0/", subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained("/data/czh/models/RealVisXL_V4.0/", subfolder="tokenizer_2")
yml_kernel = './train_kernel.yml'
train_dataset = LocalImageDataset(
    dehazing_file=[dehazing_file_path, lq_dehazing_file_path, dehazing_json_file_path],
    # deraining_file=[deraining_file_path, lq_deraining_file_path, deraining_json_file_path],
    # denoising_file=[denoising_file_path, lq_denoising_file_path, denoising_json_file_path],
    # deblurring_file=[deblurring_file_path, lq_deblurring_file_path, deblurring_json_file_path],
    # low_light_file=[low_light_file_path, lq_low_light_file_path, low_light_json_file_path],
    yml_kernel=yml_kernel, image_size=512, tokenizer=tokenizer,
    tokenizer_2=tokenizer_2, t_drop_rate=0.2)
