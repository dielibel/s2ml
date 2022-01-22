#@title **Execute Image Upscaling on All Images in a Path**
import os.path as osp
import glob
import cv2
import numpy as np
import torch
from ESRGAN import RRDBNet_arch as arch
import requests
import imageio
import requests
import warnings

warnings.filterwarnings("ignore")
from google.colab import files

Choose_device = "cuda" 

model_path = 'models/RRDB_PSNR_x4.pth' #@param ['models/RRDB_ESRGAN_x4.pth','models/RRDB_PSNR_x4.pth','models/PPON_G.pth','models/PPON_D.pth']  
device = torch.device(Choose_device) 

model_path = ESRGAN_path + '/' + model_path
esr_target_directory =  "your path in quotes"#@param string

test_img_folder = esr_target_directory

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
# If results folder in ESRGAN_path doesn't exist, make one
path_tmp = ESRGAN_path + "/results"
if not os.path.exists(path_tmp):
        os.mkdir(path_tmp)

for filename in os.listdir(test_img_folder):
    filename = test_img_folder + "/" + filename
    idx += 1
    base = osp.splitext(osp.basename(filename))[0]
    print(idx, base)
    # read images
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [0, 1, 2]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    imageio.imwrite('ESRGAN/results/{:s}.png'.format(base), output.astype(np.uint8))