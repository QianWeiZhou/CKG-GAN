import torch
import module1
import module2
import torchvision.transforms as tf
from torch.utils.data import DataLoader
import numpy as np
import progressbar
import os
import argparse
# import pretty_errors
from numpy import *


def load_kd_model():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--module1_resume_path',
        default='./models/kd_module1.pth.tar',
        type=str,
        help=
        'Path for module1 resume model.'
    )
    parser.add_argument(
        '--module2_resume_path',
        default='./models/kd_module2.pth.tar',
        type=str,
        help=
        'Path for module2 resume model.'
    )

    args = parser.parse_args()

    model1 = module1.generate_model1()
    model1 = model1.cuda()
    model2 = module2.generate_model2()
    model2 = model2.cuda()

    checkpoint1 = torch.load(args.module1_resume_path)
    model1.load_state_dict(checkpoint1['state_dict'])
    checkpoint2 = torch.load(args.module2_resume_path)
    model2.load_state_dict(checkpoint2['state_dict'])
    # print(model2)
    return model1, model2