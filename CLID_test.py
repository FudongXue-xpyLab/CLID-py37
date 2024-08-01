# coding=gbk
import os
import sys
import torch
from models.unet import UnetN2N
from utils.misc import fluore_to_tensor, to_numpy, module_size
from utils.file_pare import get_mini_size, opentiff, mkdirs, pad_image, crop_back, getFilePath, checkFile
import numpy as np
from PIL import Image
import argparse
from argparse import Namespace
import json
import time
import matplotlib.pyplot as plt
import utils.deconv as dcv

plt.switch_backend('Qt5Agg')  # pip install pyqt5


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain-dir', default='./experiments', type=str, help='dir to pre-trained model')
parser.add_argument('--data-root', default=r'./images', type=str, help='dir to dataset') 
parser.add_argument('--cuda', action='store_true', default=True, help='use GPU or not, default using GPU')
parser.add_argument('--test-data', default='100nmbeads', type=str, help='data use to training')
parser.add_argument('--multi_cells', default=True, type=str, help='data use to training')  # 是否搜索相近词
parser.add_argument('--image_types', default='', type=str, help='image type')  # save code ID
parser.add_argument('--offset', default=99.0, type=int, help='offset of each data')
parser.add_argument('--scale', default=2, type=int, help='upsample of deconvolution')
args_test = parser.parse_args()
iters = [10, 20, 30, 50, 100, 200, 500, 1000, 2000]
meth = 'lanczos'  # lanczos, cubic, linear, freq
device = 'cpu' if args_test.cuda else 'cuda'
test_data = args_test.test_data
run_dir = args_test.pretrain_dir

with open(run_dir + '/args.txt') as args_file:
    args = Namespace(**json.load(args_file))
data_dir = args_test.data_root
pixel = args.pixelsize
pixel = pixel * 10 ** (-9)
lmda = 10 ** (-9) * args.laser
NA = args.NA
rootDir = data_dir
if args_test.multi_cells:
    files, test_data = getFilePath(rootDir, test_data, ['.png', '.tif'])
else:
    files, test_data = checkFile(rootDir, test_data, ['.png', '.tif'])

if files == []:
    print('No file have found! Exit!')
    sys.exit(0)
else:
    save_file = test_data + '_scale' + str(args_test.scale)
    test_dir = rootDir + f'/{save_file}/psf'
    if args_test.cuda:
        deconv_dir = rootDir + f'/{save_file}/deconv'
    else:
        deconv_dir = rootDir + f'/{save_file}/deconv'
    mkdirs([test_dir, deconv_dir])

model = UnetN2N(args.in_channels, args.out_channels)

if args.debug:
    print(model)
    print(module_size(model))
model.load_state_dict(torch.load(run_dir + f'/checkpoints/model_epoch{args.epochs}.pth',
                                 map_location='cpu'))
model = model.to(device)
model.eval()

logger = {}
gtic = time.time()
k=1
psf_file = test_dir + f'/PSF_{lmda}.tif'
if os.path.exists(psf_file):
    psfkernel = np.asarray(Image.open(psf_file)).astype(np.float32)
    psfkernel = psfkernel/sum(sum(psfkernel))
    print("Load PSF.")
    k = 0

for filename in files:
    tic = time.time()
    print(filename)
    noisyfile = os.path.join(rootDir, filename)
    apathname, ext = os.path.splitext(filename)
    # frame 1
    noisy_png = opentiff(noisyfile, [1, 2])
    back_size = noisy_png[0].shape
    imsize = get_mini_size(back_size, 5)
    maxI = noisy_png[0].max()
    # noisy_norm = (noisy_png[0] - noisy_png[0].min())/(maxI - noisy_png[0].min())
    noisy_norm = (noisy_png[0] - args_test.offset)/(maxI - args_test.offset)
    noisy_norm[noisy_norm < 0] = 0
    noisy_expand = pad_image(noisy_norm, [imsize, imsize])
    noisy = noisy_expand.unsqueeze(dim=0)

    de_noised = model(noisy.to(device))
    de_noised = to_numpy(de_noised.squeeze())
    de_noised = crop_back(de_noised, back_size, expends=0)
    de_noised = (de_noised - de_noised.min()).astype(np.float_)
    norm_noised = (maxI * de_noised / de_noised.max())
    if k:
        # psfkernel = dcv.kernel(pixel/args_test.scale, lmda, NA, n=de_noised.shape[0]*args_test.scale)
        psfkernel = dcv.kernel(pixel, lmda, NA, n=de_noised.shape[0])
        k = 0
        Image.fromarray(psfkernel).save(psf_file)

    save_file = f'{deconv_dir}/CLID_{apathname}_f1'
    dcv_data = dcv.deconvlucy(norm_noised, psf=psfkernel, save_file=save_file, iters=iters, scale=args_test.scale,
                              gpu=args_test.cuda, meth=meth)

    toc = time.time()
    print(f'{toc - tic} sec')

print(f'Done in {time.time() - gtic} sec')
