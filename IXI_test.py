from __future__ import print_function
from IXI_utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, \
    calculate_NMSE
# from trainer import MUNIT_Trainer
from trainer_IXI import MUNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image
import warnings
import progressbar
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse


def get_modal_code():
    one = torch.ones(1, 50).cuda()
    zero = torch.zeros(1, 50).cuda()
    t1_onehot = torch.cat((one, zero), 1).cuda()  # [1,1,...,1,0,0,...,0]
    t2_onehot = torch.cat((zero, one), 1).cuda()  # [0,0,...,0,1,1,...,1]
    t1_noise = 0.02 * torch.randn(1, 100).cuda()
    t2_noise = 0.02 * torch.randn(1, 100).cuda()
    t1_modalcode = t1_onehot + t1_noise
    t2_modalcode = t2_onehot + t2_noise
    return t1_modalcode, t2_modalcode


def test(encode, decode, dataloader1, dataloader2):
    psnr_a2b_sum = 0.0
    psnr_b2a_sum = 0.0
    psnr_a2b_avg = 0.0
    psnr_b2a_avg = 0.0
    nmse_a2b_sum = 0.0
    nmse_b2a_sum = 0.0
    nmse_a2b_avg = 0.0
    nmse_b2a_avg = 0.0
    ssim_a2b_sum = 0.0
    ssim_b2a_sum = 0.0
    ssim_a2b_avg = 0.0
    ssim_b2a_avg = 0.0


    t1_modalcode, t2_modalcode = get_modal_code()
    t1_modalcode = t1_modalcode.unsqueeze(2)
    m_a = t1_modalcode.unsqueeze(2)
    t2_modalcode = t2_modalcode.unsqueeze(2)
    m_b = t2_modalcode.unsqueeze(2)

    number_of_entry = len(dataloader1)
    with progressbar.ProgressBar(max_value=number_of_entry) as bar:
        with torch.no_grad():
            for i, (images_a, images_b) in enumerate(zip(dataloader1, dataloader2)):
                img_a = images_a.unsqueeze(0)
                img_b = images_b.unsqueeze(0)

                img_a = images_a.numpy()
                img_b = images_b.numpy()

                images_a = images_a.cuda().detach()
                images_b = images_b.cuda().detach()

                # Start testing
                c_a, s_a = encode(images_a)
                c_b, s_b = encode(images_b)

                s_a_concat = torch.cat((s_a, m_a), 1)
                s_b_concat = torch.cat((s_b, m_b), 1)

                outputs_a2b = decode(c_a, s_b_concat)
                outputs_b2a = decode(c_b, s_a_concat)

                output_cat = torch.cat([images_a.data, outputs_a2b.data, images_b.data, outputs_b2a.data], 0)
                image_grid = vutils.make_grid(output_cat.data, nrow=2, padding=0, normalize=True)
                path = os.path.join(opts.output_folder, 'output_num{:04d}.jpg'.format(i))
                vutils.save_image(image_grid, path, padding=0, normalize=True)

                outputs_a2b = outputs_a2b.cpu().numpy()
                outputs_b2a = outputs_b2a.cpu().numpy()

                psnr_a2b = psnr(img_b, outputs_a2b)
                psnr_b2a = psnr(img_a, outputs_b2a)

                psnr_a2b_sum += psnr_a2b
                psnr_b2a_sum += psnr_b2a

                psnr_a2b_avg = psnr_a2b_sum / number_of_entry
                psnr_b2a_avg = psnr_b2a_sum / number_of_entry

                nmse_a2b = mse(img_b, outputs_a2b)
                nmse_b2a = mse(img_a, outputs_b2a)

                nmse_a2b_sum += nmse_a2b
                nmse_b2a_sum += nmse_b2a
                nmse_a2b_avg = nmse_a2b_sum / number_of_entry
                nmse_b2a_avg = nmse_b2a_sum / number_of_entry

                ssmi_imag_a = np.squeeze(img_a)
                ssmi_imag_b = np.squeeze(img_b)
                ssmi_output_a2b = np.squeeze(outputs_a2b)
                ssmi_output_b2a = np.squeeze(outputs_b2a)
                ssim_a2b = ssim(ssmi_imag_b, ssmi_output_a2b)
                ssim_b2a = ssim(ssmi_imag_a, ssmi_output_b2a)

                ssim_a2b_sum += ssim_a2b
                ssim_b2a_sum += ssim_b2a

                ssim_a2b_avg = ssim_a2b_sum / number_of_entry
                ssim_b2a_avg = ssim_b2a_sum / number_of_entry
                # print('\n')
                # print(psnr_a2b_avg, psnr_b2a_avg)

                bar.update(i)
                # return psnr_a2b_avg, psnr_b2a_avg
    print(psnr_a2b_avg, psnr_b2a_avg)
    print(nmse_a2b_avg, nmse_b2a_avg)
    print(ssim_a2b_avg, ssim_b2a_avg)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./outputs/train_IXI/config.yaml', help="net configuration")
    parser.add_argument('--output_folder', default='./test', type=str, help="output image path")
    parser.add_argument('--checkpoint', type=str,
                        default='./outputs/train_IXI/checkpoints/iterations_109999/gen_00110000.pt',
                        help="checkpoint of autoencoders")
    parser.add_argument('--style', type=str, default='', help="style image path")
    parser.add_argument('--seed', type=int, default=10, help="random seed")
    parser.add_argument('--num_style', type=int, default=1, help="number of styles to sample")
    parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
    parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
    parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
    parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
    opts = parser.parse_args()

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    # print(opts.output_folder)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    config = get_config(opts.config)

    # Setup model and data loader

    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)

    state_dict = torch.load(opts.checkpoint)
    trainer.gen.load_state_dict(state_dict['a'])
    # trainer.gen_b.load_state_dict(state_dict['b'])

    trainer.cuda()
    trainer.eval()
    encode = trainer.gen.encode
    decode = trainer.gen.decode

    _, _, test_loader_a, test_loader_b = get_all_data_loaders(config)  # test_loader_a:t1, test_loader_b:t2

    test(encode, decode, test_loader_a, test_loader_b)
