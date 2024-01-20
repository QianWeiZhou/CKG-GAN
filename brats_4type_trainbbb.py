"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from brats_4type_utils import get_all_data_loaders, prepare_sub_folder, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from brats_4type_trainer import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import numpy as np
import math
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_modal_code():
    one = torch.ones(1, 50).cuda()
    zero = torch.zeros(1, 50).cuda()
    t1_onehot = torch.cat((one, zero, zero, zero), 1).cuda()    #[1,1,...,1,0,0,...,0]
    t2_onehot = torch.cat((zero, one, zero, zero), 1).cuda()    #[0,0,...,0,1,1,...,1]
    t3_onehot = torch.cat((zero, zero, one, zero), 1).cuda()
    t4_onehot = torch.cat((zero, zero, zero, one), 1).cuda()
    t1_noise = 0.02*torch.randn(1, 200).cuda()
    t2_noise = 0.02*torch.randn(1, 200).cuda()
    t3_noise = 0.02*torch.randn(1, 200).cuda()
    t4_noise = 0.02*torch.randn(1, 200).cuda()

    t1_modalcode = t1_onehot + t1_noise
    t2_modalcode = t2_onehot + t2_noise
    t3_modalcode = t3_onehot + t3_noise
    t4_modalcode = t4_onehot + t4_noise

    t1_modalcode = t1_modalcode.unsqueeze(2)
    t1_modalcode = t1_modalcode.unsqueeze(2)

    t2_modalcode = t2_modalcode.unsqueeze(2)
    t2_modalcode = t2_modalcode.unsqueeze(2)

    t3_modalcode = t3_modalcode.unsqueeze(2)
    t3_modalcode = t3_modalcode.unsqueeze(2)

    t4_modalcode = t4_modalcode.unsqueeze(2)
    t4_modalcode = t4_modalcode.unsqueeze(2)
    return t1_modalcode, t2_modalcode, t3_modalcode, t4_modalcode

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/brats_4type_train.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
display_size = int(display_size)

# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
    
else:
    sys.exit("Only support MUNIT")
trainer.cuda()

train_loader_a, train_loader_b, train_loader_c, train_loader_d, test_loader_a, test_loader_b, test_loader_c, test_loader_d = get_all_data_loaders(config)

train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(1,display_size*40+1,40)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(1,display_size*40+1,40)]).cuda()
train_display_images_c = torch.stack([train_loader_c.dataset[i] for i in range(1,display_size*40+1,40)]).cuda()
train_display_images_d = torch.stack([train_loader_d.dataset[i] for i in range(1,display_size*40+1,40)]).cuda()

test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(1,display_size*40+1,40)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(1,display_size*40+1,40)]).cuda()
test_display_images_c = torch.stack([test_loader_c.dataset[i] for i in range(1,display_size*40+1,40)]).cuda()
test_display_images_d = torch.stack([test_loader_d.dataset[i] for i in range(1,display_size*40+1,40)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images_a, images_b, images_c, images_d) in enumerate(zip(train_loader_a, train_loader_b, train_loader_c, train_loader_d)):
        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        images_c, images_d = images_c.cuda().detach(), images_d.cuda().detach()
        # print(torch.min(images_a), torch.max(images_a))     # [-1, 1]
        # print(torch.min(images_b), torch.max(images_b))

        with Timer("Elapsed time in update: %f"):
            # Main training code
            t1_modalcode, t2_modalcode, t3_modalcode, t4_modalcode = get_modal_code()
            trainer.dis1_update(images_a, images_b, config, t1_modalcode, t2_modalcode)
            trainer.gen_update(images_a, images_b, config, t1_modalcode, t2_modalcode, 1)

            trainer.dis2_update(images_a, images_c, config, t1_modalcode, t3_modalcode)
            trainer.gen_update(images_a, images_c, config, t1_modalcode, t3_modalcode, 2)

            trainer.dis3_update(images_a, images_d, config, t1_modalcode, t4_modalcode)
            trainer.gen_update(images_a, images_d, config, t1_modalcode, t4_modalcode, 3)

            trainer.dis4_update(images_b, images_c, config, t2_modalcode, t3_modalcode)
            trainer.gen_update(images_b, images_c, config, t2_modalcode, t3_modalcode, 4)

            trainer.dis5_update(images_b, images_d, config, t2_modalcode, t4_modalcode)
            trainer.gen_update(images_b, images_d, config, t2_modalcode, t4_modalcode, 5)

            trainer.dis6_update(images_c, images_d, config, t3_modalcode, t4_modalcode)
            trainer.gen_update(images_c, images_d, config, t3_modalcode, t4_modalcode, 6)

            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs1 = trainer.sample1(test_display_images_a, test_display_images_b, t1_modalcode, t2_modalcode)
                test_image_outputs2 = trainer.sample2(test_display_images_a, test_display_images_c, t1_modalcode, t3_modalcode)
                test_image_outputs3 = trainer.sample3(test_display_images_a, test_display_images_d, t1_modalcode, t4_modalcode)
                test_image_outputs4 = trainer.sample4(test_display_images_b, test_display_images_c, t2_modalcode, t3_modalcode)
                test_image_outputs5 = trainer.sample5(test_display_images_b, test_display_images_d, t2_modalcode, t4_modalcode)
                test_image_outputs6 = trainer.sample6(test_display_images_c, test_display_images_d, t3_modalcode, t4_modalcode)
                # train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs1, display_size, image_directory, 'test1_%08d' % (iterations + 1))
            write_2images(test_image_outputs2, display_size, image_directory, 'test2_%08d' % (iterations + 1))
            write_2images(test_image_outputs3, display_size, image_directory, 'test3_%08d' % (iterations + 1))
            write_2images(test_image_outputs4, display_size, image_directory, 'test4_%08d' % (iterations + 1))
            write_2images(test_image_outputs5, display_size, image_directory, 'test5_%08d' % (iterations + 1))
            write_2images(test_image_outputs6, display_size, image_directory, 'test6_%08d' % (iterations + 1))
            # write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            
            # HTML
            # write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs1 = trainer.sample1(train_display_images_a, train_display_images_b, t1_modalcode, t2_modalcode)
                image_outputs2 = trainer.sample2(train_display_images_a, train_display_images_c, t1_modalcode, t3_modalcode)
                image_outputs3 = trainer.sample3(train_display_images_a, train_display_images_d, t1_modalcode, t4_modalcode)
                image_outputs4 = trainer.sample4(train_display_images_b, train_display_images_c, t2_modalcode, t3_modalcode)
                image_outputs5 = trainer.sample5(train_display_images_b, train_display_images_d, t2_modalcode, t4_modalcode)
                image_outputs6 = trainer.sample6(train_display_images_c, train_display_images_d, t3_modalcode, t4_modalcode)
                
            write_2images(image_outputs1, display_size, image_directory, 'train_current1')
            write_2images(image_outputs2, display_size, image_directory, 'train_current2')
            write_2images(image_outputs3, display_size, image_directory, 'train_current3')
            write_2images(image_outputs4, display_size, image_directory, 'train_current4')
            write_2images(image_outputs5, display_size, image_directory, 'train_current5')
            write_2images(image_outputs6, display_size, image_directory, 'train_current6')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)
        
        iterations += 1
        if iterations >= max_iter:

            sys.exit('Finish training')