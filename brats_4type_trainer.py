from brats_4type_networks import AdaINGen, MsImageDis
from brats_4type_utils import weights_init, get_model_list, kd_model_preprocess, get_scheduler, load_shapeNet
from load_kd_module import load_kd_model
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import torchvision.transforms as tf
from torchvision import transforms
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr_g = hyperparameters['lr']
        lr_d = hyperparameters['lr_dis']
        # Initiate the networks
        self.gen = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.dis_c = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_d = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        # 使用正则化的方式
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        # style 输出的特征码维度
        self.style_dim = hyperparameters['gen']['style_dim']

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        dis_params_a = list(self.dis_a.parameters())#鉴别模型a的相关参数
        dis_params_b = list(self.dis_b.parameters())
        dis_params_c = list(self.dis_c.parameters())
        dis_params_d = list(self.dis_d.parameters())
        gen_params = list(self.gen.parameters())# 生成模型的相关参数
        self.dis_opt_a = torch.optim.Adam([p for p in dis_params_a if p.requires_grad],
                                        lr=lr_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_opt_b = torch.optim.Adam([p for p in dis_params_b if p.requires_grad],
                                        lr=lr_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_opt_c = torch.optim.Adam([p for p in dis_params_c if p.requires_grad],
                                        lr=lr_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_opt_d = torch.optim.Adam([p for p in dis_params_d if p.requires_grad],
                                        lr=lr_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr_g, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler_a = get_scheduler(self.dis_opt_a, hyperparameters)
        self.dis_scheduler_b = get_scheduler(self.dis_opt_b, hyperparameters)
        self.dis_scheduler_c = get_scheduler(self.dis_opt_c, hyperparameters)
        self.dis_scheduler_d = get_scheduler(self.dis_opt_d, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization,网络模型权重初始化
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))
        self.dis_c.apply(weights_init('gaussian'))
        self.dis_d.apply(weights_init('gaussian'))

        # attention module #注意力机制
        self.gap_fc = nn.Linear(256, 1, bias=False)
        self.gmp_fc = nn.Linear(256, 1, bias=False)
        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # 计算感知 loss
        if 'per_w' in hyperparameters.keys() and hyperparameters['per_w'] > 0:

            self.kd_module1, self.kd_module2 = load_kd_model()

            for param in self.kd_module1.parameters():
                param.requires_grad = False
            for param in self.kd_module2.parameters():
                param.requires_grad = False


        if 'shape_w' in hyperparameters.keys() and hyperparameters['shape_w'] > 0:
            self.shapeNet = load_shapeNet()
            self.shapeNet.eval()
            for param in self.shapeNet.parameters():
                param.requires_grad = False


    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))


    def calc_modal_loss(self, input, target):
        return torch.sum(torch.abs(input - target))


    def forward(self, x_a, x_b, t1_code, t2_code):

        self.eval()
        # input image encode, get cotent and style code
        c_a, s_a_fake = self.gen.encode(x_a)
        c_b, s_b_fake = self.gen.encode(x_b)

        s_a_concat = torch.cat((s_a_fake, t1_code), 1)
        s_b_concat = torch.cat((s_b_fake, t2_code), 1)
        # print(s_a_concat.size())

        # b-content and a-style fusion
        x_ba = self.gen.decode(c_b, s_a_concat)
        x_ab = self.gen.decode(c_a, s_b_concat)
        self.train()
        return x_ab, x_ba

    # 生成模型进行优化
    def gen_update(self, x_a, x_b, hyperparameters, t1_code, t2_code, type_num):
    	# 给输入的 x_a, x_b 加入随机噪声
        self.gen_opt.zero_grad()

        # a, b 图像进行编码
        c_a, s_a_prime = self.gen.encode(x_a)
        c_b, s_b_prime = self.gen.encode(x_b)

        s_a_concat = torch.cat((s_a_prime, t1_code), 1)
        s_b_concat = torch.cat((s_b_prime, t2_code), 1)
        # print(s_a_concat.size())

        # attention
        gap_a = torch.nn.functional.adaptive_avg_pool2d(c_a, 1) # [1, 256, 1, 1]
        gap_logit_a = self.gap_fc(gap_a.view(c_a.shape[0], -1)) # [1, 1]
        gap_weight_a = list(self.gap_fc.parameters())[0]        # [1, 256]
        gap_a = c_a * gap_weight_a.unsqueeze(2).unsqueeze(3)    # [1, 256, 64, 64]

        gap_b = torch.nn.functional.adaptive_avg_pool2d(c_b, 1)
        gap_logit_b = self.gap_fc(gap_b.view(c_b.shape[0], -1))
        gap_weight_b = list(self.gap_fc.parameters())[0]
        gap_b = c_b * gap_weight_b.unsqueeze(2).unsqueeze(3)

        gmp_a = torch.nn.functional.adaptive_max_pool2d(c_a, 1)
        gmp_logit_a = self.gmp_fc(gmp_a.view(c_a.shape[0], -1))
        gmp_weight_a = list(self.gmp_fc.parameters())[0]
        gmp_a = c_a * gmp_weight_a.unsqueeze(2).unsqueeze(3)

        gmp_b = torch.nn.functional.adaptive_max_pool2d(c_b, 1)
        gmp_logit_b = self.gmp_fc(gmp_b.view(c_b.shape[0], -1))
        gmp_weight_b = list(self.gmp_fc.parameters())[0]
        gmp_b = c_b * gmp_weight_b.unsqueeze(2).unsqueeze(3)

        x_a_attetion = torch.cat([gap_a, gmp_a], 1)             # [1, 512, 64, 64]
        c_a_attetion = self.relu(self.conv1x1(x_a_attetion))    # [1, 256, 64, 64]

        x_b_attetion = torch.cat([gap_b, gmp_b], 1)
        c_b_attetion = self.relu(self.conv1x1(x_b_attetion))


        # print(s_concat.size())
        # decode (within domain)
        x_a_recon = self.gen.decode(c_a, s_a_concat)
        x_b_recon = self.gen.decode(c_b, s_b_concat)

        # decode (cross domain),进行交叉解码，即两张图片的content code，style code进行互换
        x_ab = self.gen.decode(c_a_attetion, s_b_concat)
        x_ba = self.gen.decode(c_b_attetion, s_a_concat)
        # print(torch.min(x_ba), torch.max(x_ba))


        # encode again,对上面合成的图片再进行编码，得到重构的content code，style code
        c_a_recon, s_b_recon = self.gen.encode(x_ab)
        c_b_recon, s_a_recon = self.gen.encode(x_ba)

        s_a_concat_recon = torch.cat((s_a_recon, t1_code), 1)
        s_b_concat_recon = torch.cat((s_b_recon, t2_code), 1)


        # decode again (if needed),重构的content code 与真实图片编码得到 style code（s_x_prime）进行解码，生成新图片
        x_axa = self.gen.decode(c_a_recon, s_a_concat_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bxb = self.gen.decode(c_b_recon, s_b_concat_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss,重构图片a and b，与真实图片计算loss
        # print(x_a_recon.size(), x_a.size())
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        # 合成图片，再编码得到的style code，与s_x（style code）计算loss
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a_prime)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b_prime)
        # 由合成图片编码得到的content code，与真实的图片编码得到的content code 计算loss
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)

        # 循环一致性loss，计算x_aba 与 x_a 的loss
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_axa, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bxb, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
  
        
        # GAN loss,最终生成图片与真实图片之间的loss
        if type_num == 1:
            self.loss_gen_adv_ba = self.dis_a.calc_gen_loss(x_ba)
            self.loss_gen_adv_ab = self.dis_b.calc_gen_loss(x_ab)
        elif type_num == 2:
            self.loss_gen_adv_ca = self.dis_a.calc_gen_loss(x_ba)
            self.loss_gen_adv_ac = self.dis_c.calc_gen_loss(x_ab)
        elif type_num == 3:
            self.loss_gen_adv_da = self.dis_a.calc_gen_loss(x_ba)
            self.loss_gen_adv_ad = self.dis_d.calc_gen_loss(x_ab)
        elif type_num == 4:
            self.loss_gen_adv_cb = self.dis_b.calc_gen_loss(x_ba)
            self.loss_gen_adv_bc = self.dis_c.calc_gen_loss(x_ab)
        elif type_num == 5:
            self.loss_gen_adv_db = self.dis_b.calc_gen_loss(x_ba)
            self.loss_gen_adv_bd = self.dis_d.calc_gen_loss(x_ab)
        elif type_num == 6:
            self.loss_gen_adv_dc = self.dis_c.calc_gen_loss(x_ba)
            self.loss_gen_adv_cd = self.dis_d.calc_gen_loss(x_ab)
        # perceptual loss,使用kd_model计算感知loss
        self.loss_per_ab = self.compute_perception_loss(self.kd_module1, self.kd_module2, x_ab, x_a) if hyperparameters['per_w'] > 0 else 0
        self.loss_per_ba = self.compute_perception_loss(self.kd_module1, self.kd_module2, x_ba, x_b) if hyperparameters['per_w'] > 0 else 0
    
        # shape-loss
        x_ab = x_ab.expand(-1, 3, -1, -1)
        x_ba = x_ba.expand(-1, 3, -1, -1)

        x_a = x_a.expand(-1, 3, -1, -1)
        x_b = x_b.expand(-1, 3, -1, -1)


        self.loss_shape_ab = self.compute_shape_loss(self.shapeNet, x_ab, x_a) if hyperparameters['shape_w'] > 0 else 0
        self.loss_shape_ba = self.compute_shape_loss(self.shapeNet, x_ba, x_b) if hyperparameters['shape_w'] > 0 else 0
        
        # total loss
        if type_num == 1:
            self.loss_gen_total =   hyperparameters['gan_w'] * self.loss_gen_adv_ab + \
                                    hyperparameters['gan_w'] * self.loss_gen_adv_ba + \
                                    hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                                    hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                                    hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                                    hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                                    hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                                    hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                                    hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                                    hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                                    hyperparameters['per_w'] * self.loss_per_ab + \
                                    hyperparameters['per_w'] * self.loss_per_ba + \
                                    hyperparameters['shape_w'] * self.loss_shape_ab + \
                                    hyperparameters['shape_w'] * self.loss_shape_ba
        elif type_num == 2:
            self.loss_gen_total =   hyperparameters['gan_w'] * self.loss_gen_adv_ca + \
                                    hyperparameters['gan_w'] * self.loss_gen_adv_ac + \
                                    hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                                    hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                                    hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                                    hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                                    hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                                    hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                                    hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                                    hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                                    hyperparameters['per_w'] * self.loss_per_ab + \
                                    hyperparameters['per_w'] * self.loss_per_ba + \
                                    hyperparameters['shape_w'] * self.loss_shape_ab + \
                                    hyperparameters['shape_w'] * self.loss_shape_ba
        elif type_num == 3:
            self.loss_gen_total =   hyperparameters['gan_w'] * self.loss_gen_adv_da + \
                                    hyperparameters['gan_w'] * self.loss_gen_adv_ad + \
                                    hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                                    hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                                    hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                                    hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                                    hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                                    hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                                    hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                                    hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                                    hyperparameters['per_w'] * self.loss_per_ab + \
                                    hyperparameters['per_w'] * self.loss_per_ba + \
                                    hyperparameters['shape_w'] * self.loss_shape_ab + \
                                    hyperparameters['shape_w'] * self.loss_shape_ba
        elif type_num == 4:
            self.loss_gen_total =   hyperparameters['gan_w'] * self.loss_gen_adv_cb + \
                                    hyperparameters['gan_w'] * self.loss_gen_adv_bc + \
                                    hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                                    hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                                    hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                                    hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                                    hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                                    hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                                    hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                                    hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                                    hyperparameters['per_w'] * self.loss_per_ab + \
                                    hyperparameters['per_w'] * self.loss_per_ba + \
                                    hyperparameters['shape_w'] * self.loss_shape_ab + \
                                    hyperparameters['shape_w'] * self.loss_shape_ba
        elif type_num == 5:
            self.loss_gen_total =   hyperparameters['gan_w'] * self.loss_gen_adv_db + \
                                    hyperparameters['gan_w'] * self.loss_gen_adv_bd + \
                                    hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                                    hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                                    hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                                    hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                                    hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                                    hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                                    hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                                    hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                                    hyperparameters['per_w'] * self.loss_per_ab + \
                                    hyperparameters['per_w'] * self.loss_per_ba + \
                                    hyperparameters['shape_w'] * self.loss_shape_ab + \
                                    hyperparameters['shape_w'] * self.loss_shape_ba
        elif type_num == 6:
            self.loss_gen_total =   hyperparameters['gan_w'] * self.loss_gen_adv_dc + \
                                    hyperparameters['gan_w'] * self.loss_gen_adv_cd + \
                                    hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                                    hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                                    hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                                    hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                                    hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                                    hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                                    hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                                    hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                                    hyperparameters['per_w'] * self.loss_per_ab + \
                                    hyperparameters['per_w'] * self.loss_per_ba + \
                                    hyperparameters['shape_w'] * self.loss_shape_ab + \
                                    hyperparameters['shape_w'] * self.loss_shape_ba

        self.loss_gen_total.backward()
        self.gen_opt.step()


    def compute_perception_loss(self, kd_module1, kd_module2, img, target):
        img_kd = kd_model_preprocess(img)
        target_kd = kd_model_preprocess(target)
        img_fea = kd_module1(img_kd)
        img_fea = kd_module2(img_fea)
        target_fea = kd_module1(target_kd)
        target_fea = kd_module2(target_fea)
        # print(img_fea.size())
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def compute_shape_loss(self, shapenet, img, target):
        # print(img.size())
        img_shape = shapenet(img)
        target_shape = shapenet(target)
        return torch.mean((self.instancenorm(img_shape) - self.instancenorm(target_shape)) ** 2)
    

    def update_learning_rate(self):
        if self.dis_scheduler_a is not None:
            self.dis_scheduler_a.step()
        if self.dis_scheduler_b is not None:
            self.dis_scheduler_b.step()
        if self.dis_scheduler_c is not None:
            self.dis_scheduler_c.step()
        if self.dis_scheduler_d is not None:
            self.dis_scheduler_d.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        #opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict(), 'c': self.dis_c.state_dict(), 'd': self.dis_d.state_dict()}, dis_name)
        #torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    
    # 鉴别模型进行优化
    def dis1_update(self, x_a, x_b, hyperparameters, t1_code, t2_code):
        self.dis_opt_a.zero_grad()
        self.dis_opt_b.zero_grad()

        c_a, s_a = self.gen.encode(x_a)
        c_b, s_b = self.gen.encode(x_b)

        s_a_concat = torch.cat((s_a, t1_code), 1)
        # print(s_a_concat.size())
        s_b_concat = torch.cat((s_b, t2_code), 1)

        # attention
        gap_a = torch.nn.functional.adaptive_avg_pool2d(c_a, 1) # [1, 256, 1, 1]
        gap_logit_a = self.gap_fc(gap_a.view(c_a.shape[0], -1)) # [1, 1]
        gap_weight_a = list(self.gap_fc.parameters())[0]        # [1, 256]
        gap_a = c_a * gap_weight_a.unsqueeze(2).unsqueeze(3)    # [1, 256, 64, 64]

        gap_b = torch.nn.functional.adaptive_avg_pool2d(c_b, 1)
        gap_logit_b = self.gap_fc(gap_b.view(c_b.shape[0], -1))
        gap_weight_b = list(self.gap_fc.parameters())[0]
        gap_b = c_b * gap_weight_b.unsqueeze(2).unsqueeze(3)

        gmp_a = torch.nn.functional.adaptive_max_pool2d(c_a, 1)
        gmp_logit_a = self.gmp_fc(gmp_a.view(c_a.shape[0], -1))
        gmp_weight_a = list(self.gmp_fc.parameters())[0]
        gmp_a = c_a * gmp_weight_a.unsqueeze(2).unsqueeze(3)

        gmp_b = torch.nn.functional.adaptive_max_pool2d(c_b, 1)
        gmp_logit_b = self.gmp_fc(gmp_b.view(c_b.shape[0], -1))
        gmp_weight_b = list(self.gmp_fc.parameters())[0]
        gmp_b = c_b * gmp_weight_b.unsqueeze(2).unsqueeze(3)

        x_a_attetion = torch.cat([gap_a, gmp_a], 1)             # [1, 512, 64, 64]
        c_a_attetion = self.relu(self.conv1x1(x_a_attetion))    # [1, 256, 64, 64]
        # print(c_a_attetion.size())
        # heatmap_a = torch.sum(c_a_attetion, dim=1, keepdim=True)

        x_b_attetion = torch.cat([gap_b, gmp_b], 1)
        c_b_attetion = self.relu(self.conv1x1(x_b_attetion))


        # decode (cross domain),交叉进行解码（即互换 content code 或者 style code）
        x_ba = self.gen.decode(c_b_attetion, s_a_concat)
        x_ab = self.gen.decode(c_a_attetion, s_b_concat)
        
        # D loss
        self.loss_dis_a = hyperparameters['gan_w'] *(self.dis_a.calc_dis_loss(x_ba.detach(), x_a))
        self.loss_dis_b = hyperparameters['gan_w'] *(self.dis_b.calc_dis_loss(x_ab.detach(), x_b))
        # self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_c + hyperparameters['gan_w'] * self.loss_dis_d
        self.loss_dis_a.backward()
        self.dis_opt_a.step() 
        self.loss_dis_b.backward() 
        self.dis_opt_b.step()

    # 鉴别模型进行优化
    def dis2_update(self, x_a, x_b, hyperparameters, t1_code, t2_code):
        self.dis_opt_a.zero_grad()
        self.dis_opt_c.zero_grad()
        c_a, s_a = self.gen.encode(x_a)
        c_b, s_b = self.gen.encode(x_b)

        s_a_concat = torch.cat((s_a, t1_code), 1)
        # print(s_a_concat.size())
        s_b_concat = torch.cat((s_b, t2_code), 1)

        # attention
        gap_a = torch.nn.functional.adaptive_avg_pool2d(c_a, 1) # [1, 256, 1, 1]
        gap_logit_a = self.gap_fc(gap_a.view(c_a.shape[0], -1)) # [1, 1]
        gap_weight_a = list(self.gap_fc.parameters())[0]        # [1, 256]
        gap_a = c_a * gap_weight_a.unsqueeze(2).unsqueeze(3)    # [1, 256, 64, 64]

        gap_b = torch.nn.functional.adaptive_avg_pool2d(c_b, 1)
        gap_logit_b = self.gap_fc(gap_b.view(c_b.shape[0], -1))
        gap_weight_b = list(self.gap_fc.parameters())[0]
        gap_b = c_b * gap_weight_b.unsqueeze(2).unsqueeze(3)

        gmp_a = torch.nn.functional.adaptive_max_pool2d(c_a, 1)
        gmp_logit_a = self.gmp_fc(gmp_a.view(c_a.shape[0], -1))
        gmp_weight_a = list(self.gmp_fc.parameters())[0]
        gmp_a = c_a * gmp_weight_a.unsqueeze(2).unsqueeze(3)

        gmp_b = torch.nn.functional.adaptive_max_pool2d(c_b, 1)
        gmp_logit_b = self.gmp_fc(gmp_b.view(c_b.shape[0], -1))
        gmp_weight_b = list(self.gmp_fc.parameters())[0]
        gmp_b = c_b * gmp_weight_b.unsqueeze(2).unsqueeze(3)

        x_a_attetion = torch.cat([gap_a, gmp_a], 1)             # [1, 512, 64, 64]
        c_a_attetion = self.relu(self.conv1x1(x_a_attetion))    # [1, 256, 64, 64]
        # print(c_a_attetion.size())
        # heatmap_a = torch.sum(c_a_attetion, dim=1, keepdim=True)

        x_b_attetion = torch.cat([gap_b, gmp_b], 1)
        c_b_attetion = self.relu(self.conv1x1(x_b_attetion))


        # decode (cross domain),交叉进行解码（即互换 content code 或者 style code）
        x_ba = self.gen.decode(c_b_attetion, s_a_concat)
        x_ab = self.gen.decode(c_a_attetion, s_b_concat)
        
        # D loss
        self.loss_dis_a = hyperparameters['gan_w'] *(self.dis_a.calc_dis_loss(x_ba.detach(), x_a))
        self.loss_dis_c = hyperparameters['gan_w'] *(self.dis_c.calc_dis_loss(x_ab.detach(), x_b))
        # self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_c + hyperparameters['gan_w'] * self.loss_dis_d
        self.loss_dis_a.backward()
        self.dis_opt_a.step() 
        self.loss_dis_c.backward() 
        self.dis_opt_c.step()


    # 鉴别模型进行优化
    def dis3_update(self, x_a, x_b, hyperparameters, t1_code, t2_code):
        self.dis_opt_a.zero_grad()
        self.dis_opt_d.zero_grad()

        c_a, s_a = self.gen.encode(x_a)
        c_b, s_b = self.gen.encode(x_b)

        s_a_concat = torch.cat((s_a, t1_code), 1)
        # print(s_a_concat.size())
        s_b_concat = torch.cat((s_b, t2_code), 1)

        # attention
        gap_a = torch.nn.functional.adaptive_avg_pool2d(c_a, 1) # [1, 256, 1, 1]
        gap_logit_a = self.gap_fc(gap_a.view(c_a.shape[0], -1)) # [1, 1]
        gap_weight_a = list(self.gap_fc.parameters())[0]        # [1, 256]
        gap_a = c_a * gap_weight_a.unsqueeze(2).unsqueeze(3)    # [1, 256, 64, 64]

        gap_b = torch.nn.functional.adaptive_avg_pool2d(c_b, 1)
        gap_logit_b = self.gap_fc(gap_b.view(c_b.shape[0], -1))
        gap_weight_b = list(self.gap_fc.parameters())[0]
        gap_b = c_b * gap_weight_b.unsqueeze(2).unsqueeze(3)

        gmp_a = torch.nn.functional.adaptive_max_pool2d(c_a, 1)
        gmp_logit_a = self.gmp_fc(gmp_a.view(c_a.shape[0], -1))
        gmp_weight_a = list(self.gmp_fc.parameters())[0]
        gmp_a = c_a * gmp_weight_a.unsqueeze(2).unsqueeze(3)

        gmp_b = torch.nn.functional.adaptive_max_pool2d(c_b, 1)
        gmp_logit_b = self.gmp_fc(gmp_b.view(c_b.shape[0], -1))
        gmp_weight_b = list(self.gmp_fc.parameters())[0]
        gmp_b = c_b * gmp_weight_b.unsqueeze(2).unsqueeze(3)

        x_a_attetion = torch.cat([gap_a, gmp_a], 1)             # [1, 512, 64, 64]
        c_a_attetion = self.relu(self.conv1x1(x_a_attetion))    # [1, 256, 64, 64]
        # print(c_a_attetion.size())
        # heatmap_a = torch.sum(c_a_attetion, dim=1, keepdim=True)

        x_b_attetion = torch.cat([gap_b, gmp_b], 1)
        c_b_attetion = self.relu(self.conv1x1(x_b_attetion))


        # decode (cross domain),交叉进行解码（即互换 content code 或者 style code）
        x_ba = self.gen.decode(c_b_attetion, s_a_concat)
        x_ab = self.gen.decode(c_a_attetion, s_b_concat)
        
        # D loss
        self.loss_dis_a = hyperparameters['gan_w'] *(self.dis_a.calc_dis_loss(x_ba.detach(), x_a))
        self.loss_dis_d = hyperparameters['gan_w'] *(self.dis_d.calc_dis_loss(x_ab.detach(), x_b))
        # self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_c + hyperparameters['gan_w'] * self.loss_dis_d
        self.loss_dis_a.backward()
        self.dis_opt_a.step() 
        self.loss_dis_d.backward() 
        self.dis_opt_d.step()

    # 鉴别模型进行优化
    def dis4_update(self, x_a, x_b, hyperparameters, t1_code, t2_code):
        self.dis_opt_b.zero_grad()
        self.dis_opt_c.zero_grad()

        c_a, s_a = self.gen.encode(x_a)
        c_b, s_b = self.gen.encode(x_b)

        s_a_concat = torch.cat((s_a, t1_code), 1)
        # print(s_a_concat.size())
        s_b_concat = torch.cat((s_b, t2_code), 1)

        # attention
        gap_a = torch.nn.functional.adaptive_avg_pool2d(c_a, 1) # [1, 256, 1, 1]
        gap_logit_a = self.gap_fc(gap_a.view(c_a.shape[0], -1)) # [1, 1]
        gap_weight_a = list(self.gap_fc.parameters())[0]        # [1, 256]
        gap_a = c_a * gap_weight_a.unsqueeze(2).unsqueeze(3)    # [1, 256, 64, 64]

        gap_b = torch.nn.functional.adaptive_avg_pool2d(c_b, 1)
        gap_logit_b = self.gap_fc(gap_b.view(c_b.shape[0], -1))
        gap_weight_b = list(self.gap_fc.parameters())[0]
        gap_b = c_b * gap_weight_b.unsqueeze(2).unsqueeze(3)

        gmp_a = torch.nn.functional.adaptive_max_pool2d(c_a, 1)
        gmp_logit_a = self.gmp_fc(gmp_a.view(c_a.shape[0], -1))
        gmp_weight_a = list(self.gmp_fc.parameters())[0]
        gmp_a = c_a * gmp_weight_a.unsqueeze(2).unsqueeze(3)

        gmp_b = torch.nn.functional.adaptive_max_pool2d(c_b, 1)
        gmp_logit_b = self.gmp_fc(gmp_b.view(c_b.shape[0], -1))
        gmp_weight_b = list(self.gmp_fc.parameters())[0]
        gmp_b = c_b * gmp_weight_b.unsqueeze(2).unsqueeze(3)

        x_a_attetion = torch.cat([gap_a, gmp_a], 1)             # [1, 512, 64, 64]
        c_a_attetion = self.relu(self.conv1x1(x_a_attetion))    # [1, 256, 64, 64]
        # print(c_a_attetion.size())
        # heatmap_a = torch.sum(c_a_attetion, dim=1, keepdim=True)

        x_b_attetion = torch.cat([gap_b, gmp_b], 1)
        c_b_attetion = self.relu(self.conv1x1(x_b_attetion))


        # decode (cross domain),交叉进行解码（即互换 content code 或者 style code）
        x_ba = self.gen.decode(c_b_attetion, s_a_concat)
        x_ab = self.gen.decode(c_a_attetion, s_b_concat)
        
        # D loss
        self.loss_dis_b = hyperparameters['gan_w'] *(self.dis_b.calc_dis_loss(x_ba.detach(), x_a))
        self.loss_dis_c = hyperparameters['gan_w'] *(self.dis_c.calc_dis_loss(x_ab.detach(), x_b))
        # self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_c + hyperparameters['gan_w'] * self.loss_dis_d
        self.loss_dis_b.backward()
        self.dis_opt_b.step() 
        self.loss_dis_c.backward()
        self.dis_opt_c.step()

    # 鉴别模型进行优化
    def dis5_update(self, x_a, x_b, hyperparameters, t1_code, t2_code):
        self.dis_opt_b.zero_grad()
        self.dis_opt_d.zero_grad()

        c_a, s_a = self.gen.encode(x_a)
        c_b, s_b = self.gen.encode(x_b)

        s_a_concat = torch.cat((s_a, t1_code), 1)
        # print(s_a_concat.size())
        s_b_concat = torch.cat((s_b, t2_code), 1)

        # attention
        gap_a = torch.nn.functional.adaptive_avg_pool2d(c_a, 1) # [1, 256, 1, 1]
        gap_logit_a = self.gap_fc(gap_a.view(c_a.shape[0], -1)) # [1, 1]
        gap_weight_a = list(self.gap_fc.parameters())[0]        # [1, 256]
        gap_a = c_a * gap_weight_a.unsqueeze(2).unsqueeze(3)    # [1, 256, 64, 64]

        gap_b = torch.nn.functional.adaptive_avg_pool2d(c_b, 1)
        gap_logit_b = self.gap_fc(gap_b.view(c_b.shape[0], -1))
        gap_weight_b = list(self.gap_fc.parameters())[0]
        gap_b = c_b * gap_weight_b.unsqueeze(2).unsqueeze(3)

        gmp_a = torch.nn.functional.adaptive_max_pool2d(c_a, 1)
        gmp_logit_a = self.gmp_fc(gmp_a.view(c_a.shape[0], -1))
        gmp_weight_a = list(self.gmp_fc.parameters())[0]
        gmp_a = c_a * gmp_weight_a.unsqueeze(2).unsqueeze(3)

        gmp_b = torch.nn.functional.adaptive_max_pool2d(c_b, 1)
        gmp_logit_b = self.gmp_fc(gmp_b.view(c_b.shape[0], -1))
        gmp_weight_b = list(self.gmp_fc.parameters())[0]
        gmp_b = c_b * gmp_weight_b.unsqueeze(2).unsqueeze(3)

        x_a_attetion = torch.cat([gap_a, gmp_a], 1)             # [1, 512, 64, 64]
        c_a_attetion = self.relu(self.conv1x1(x_a_attetion))    # [1, 256, 64, 64]
        # print(c_a_attetion.size())
        # heatmap_a = torch.sum(c_a_attetion, dim=1, keepdim=True)

        x_b_attetion = torch.cat([gap_b, gmp_b], 1)
        c_b_attetion = self.relu(self.conv1x1(x_b_attetion))


        # decode (cross domain),交叉进行解码（即互换 content code 或者 style code）
        x_ba = self.gen.decode(c_b_attetion, s_a_concat)
        x_ab = self.gen.decode(c_a_attetion, s_b_concat)
        
        # D loss
        self.loss_dis_b = hyperparameters['gan_w'] *(self.dis_b.calc_dis_loss(x_ba.detach(), x_a))
        self.loss_dis_d = hyperparameters['gan_w'] *(self.dis_d.calc_dis_loss(x_ab.detach(), x_b))
        # self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_c + hyperparameters['gan_w'] * self.loss_dis_d
        self.loss_dis_b.backward()
        self.dis_opt_b.step() 
        self.loss_dis_d.backward() 
        self.dis_opt_d.step()

    # 鉴别模型进行优化
    def dis6_update(self, x_a, x_b, hyperparameters, t1_code, t2_code):
        self.dis_opt_c.zero_grad()
        self.dis_opt_d.zero_grad()

        c_a, s_a = self.gen.encode(x_a)
        c_b, s_b = self.gen.encode(x_b)

        s_a_concat = torch.cat((s_a, t1_code), 1)
        # print(s_a_concat.size())
        s_b_concat = torch.cat((s_b, t2_code), 1)

        # attention
        gap_a = torch.nn.functional.adaptive_avg_pool2d(c_a, 1) # [1, 256, 1, 1]
        gap_logit_a = self.gap_fc(gap_a.view(c_a.shape[0], -1)) # [1, 1]
        gap_weight_a = list(self.gap_fc.parameters())[0]        # [1, 256]
        gap_a = c_a * gap_weight_a.unsqueeze(2).unsqueeze(3)    # [1, 256, 64, 64]

        gap_b = torch.nn.functional.adaptive_avg_pool2d(c_b, 1)
        gap_logit_b = self.gap_fc(gap_b.view(c_b.shape[0], -1))
        gap_weight_b = list(self.gap_fc.parameters())[0]
        gap_b = c_b * gap_weight_b.unsqueeze(2).unsqueeze(3)

        gmp_a = torch.nn.functional.adaptive_max_pool2d(c_a, 1)
        gmp_logit_a = self.gmp_fc(gmp_a.view(c_a.shape[0], -1))
        gmp_weight_a = list(self.gmp_fc.parameters())[0]
        gmp_a = c_a * gmp_weight_a.unsqueeze(2).unsqueeze(3)

        gmp_b = torch.nn.functional.adaptive_max_pool2d(c_b, 1)
        gmp_logit_b = self.gmp_fc(gmp_b.view(c_b.shape[0], -1))
        gmp_weight_b = list(self.gmp_fc.parameters())[0]
        gmp_b = c_b * gmp_weight_b.unsqueeze(2).unsqueeze(3)

        x_a_attetion = torch.cat([gap_a, gmp_a], 1)             # [1, 512, 64, 64]
        c_a_attetion = self.relu(self.conv1x1(x_a_attetion))    # [1, 256, 64, 64]
        # print(c_a_attetion.size())
        # heatmap_a = torch.sum(c_a_attetion, dim=1, keepdim=True)

        x_b_attetion = torch.cat([gap_b, gmp_b], 1)
        c_b_attetion = self.relu(self.conv1x1(x_b_attetion))


        # decode (cross domain),交叉进行解码（即互换 content code 或者 style code）
        x_ba = self.gen.decode(c_b_attetion, s_a_concat)
        x_ab = self.gen.decode(c_a_attetion, s_b_concat)
        
        # D loss
        self.loss_dis_c = hyperparameters['gan_w'] *(self.dis_c.calc_dis_loss(x_ba.detach(), x_a))
        self.loss_dis_d = hyperparameters['gan_w'] *(self.dis_d.calc_dis_loss(x_ab.detach(), x_b))
        # self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_c + hyperparameters['gan_w'] * self.loss_dis_d
        self.loss_dis_c.backward()
        self.dis_opt_c.step() 
        self.loss_dis_d.backward() 
        self.dis_opt_d.step()

            # 训练的时候，查看目前的效果
    def sample1(self, x_a, x_b, t1_code, t2_code):
        self.eval()

        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []

        for i in range(x_a.size(0)):
            # 输入图片a,b，分别得到对应的content code 以及 style code
            c_a, s_a_fake = self.gen.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0))

            s_a_concat = torch.cat((s_a_fake, t1_code), 1)
            s_b_concat = torch.cat((s_b_fake, t2_code), 1)
            # 对图片a进行重构，使用从图片a分离出来的content code 以及 style code
            x_a_recon.append(self.gen.decode(c_a, s_a_concat))
            # 对图片b进行重构，使用从图片b分离出来的content code 以及 style code
            x_b_recon.append(self.gen.decode(c_b, s_b_concat))

            # 使用分离出来的content code， 结合符合正太分布随机生成的style code，生成图片
            x_ba.append(self.gen.decode(c_b, s_a_concat.unsqueeze(0)))
            x_ab.append(self.gen.decode(c_a, s_b_concat.unsqueeze(0)))

        # 把图片的像素连接起来
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

        # 训练的时候，查看目前的效果
    def sample2(self, x_a, x_b, t1_code, t2_code):
        self.eval()

        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []

        for i in range(x_a.size(0)):
            # 输入图片a,b，分别得到对应的content code 以及 style code
            c_a, s_a_fake = self.gen.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0))

            s_a_concat = torch.cat((s_a_fake, t1_code), 1)
            s_b_concat = torch.cat((s_b_fake, t2_code), 1)
            # 对图片a进行重构，使用从图片a分离出来的content code 以及 style code
            x_a_recon.append(self.gen.decode(c_a, s_a_concat))
            # 对图片b进行重构，使用从图片b分离出来的content code 以及 style code
            x_b_recon.append(self.gen.decode(c_b, s_b_concat))

            # 使用分离出来的content code， 结合符合正太分布随机生成的style code，生成图片
            x_ba.append(self.gen.decode(c_b, s_a_concat.unsqueeze(0)))
            x_ab.append(self.gen.decode(c_a, s_b_concat.unsqueeze(0)))
            

        # 把图片的像素连接起来
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

        # 训练的时候，查看目前的效果
    def sample3(self, x_a, x_b, t1_code, t2_code):
        self.eval()

        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []

        for i in range(x_a.size(0)):
            # 输入图片a,b，分别得到对应的content code 以及 style code
            c_a, s_a_fake = self.gen.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0))

            s_a_concat = torch.cat((s_a_fake, t1_code), 1)
            s_b_concat = torch.cat((s_b_fake, t2_code), 1)
            # 对图片a进行重构，使用从图片a分离出来的content code 以及 style code
            x_a_recon.append(self.gen.decode(c_a, s_a_concat))
            # 对图片b进行重构，使用从图片b分离出来的content code 以及 style code
            x_b_recon.append(self.gen.decode(c_b, s_b_concat))

            # 使用分离出来的content code， 结合符合正太分布随机生成的style code，生成图片
            x_ba.append(self.gen.decode(c_b, s_a_concat.unsqueeze(0)))
            x_ab.append(self.gen.decode(c_a, s_b_concat.unsqueeze(0)))
            

        # 把图片的像素连接起来
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

        # 训练的时候，查看目前的效果
    def sample4(self, x_a, x_b, t1_code, t2_code):
        self.eval()

        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []

        for i in range(x_a.size(0)):
            # 输入图片a,b，分别得到对应的content code 以及 style code
            c_a, s_a_fake = self.gen.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0))

            s_a_concat = torch.cat((s_a_fake, t1_code), 1)
            s_b_concat = torch.cat((s_b_fake, t2_code), 1)
            # 对图片a进行重构，使用从图片a分离出来的content code 以及 style code
            x_a_recon.append(self.gen.decode(c_a, s_a_concat))
            # 对图片b进行重构，使用从图片b分离出来的content code 以及 style code
            x_b_recon.append(self.gen.decode(c_b, s_b_concat))

            # 使用分离出来的content code， 结合符合正太分布随机生成的style code，生成图片
            x_ba.append(self.gen.decode(c_b, s_a_concat.unsqueeze(0)))
            x_ab.append(self.gen.decode(c_a, s_b_concat.unsqueeze(0)))
            

        # 把图片的像素连接起来
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def sample5(self, x_a, x_b, t1_code, t2_code):
        self.eval()

        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []

        for i in range(x_a.size(0)):
            # 输入图片a,b，分别得到对应的content code 以及 style code
            c_a, s_a_fake = self.gen.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0))

            s_a_concat = torch.cat((s_a_fake, t1_code), 1)
            s_b_concat = torch.cat((s_b_fake, t2_code), 1)
            # 对图片a进行重构，使用从图片a分离出来的content code 以及 style code
            x_a_recon.append(self.gen.decode(c_a, s_a_concat))
            # 对图片b进行重构，使用从图片b分离出来的content code 以及 style code
            x_b_recon.append(self.gen.decode(c_b, s_b_concat))

            # 使用分离出来的content code， 结合符合正太分布随机生成的style code，生成图片
            x_ba.append(self.gen.decode(c_b, s_a_concat.unsqueeze(0)))
            x_ab.append(self.gen.decode(c_a, s_b_concat.unsqueeze(0)))
            

        # 把图片的像素连接起来
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def sample6(self, x_a, x_b, t1_code, t2_code):
        self.eval()

        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []

        for i in range(x_a.size(0)):
            # 输入图片a,b，分别得到对应的content code 以及 style code
            c_a, s_a_fake = self.gen.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0))

            s_a_concat = torch.cat((s_a_fake, t1_code), 1)
            s_b_concat = torch.cat((s_b_fake, t2_code), 1)
            # 对图片a进行重构，使用从图片a分离出来的content code 以及 style code
            x_a_recon.append(self.gen.decode(c_a, s_a_concat))
            # 对图片b进行重构，使用从图片b分离出来的content code 以及 style code
            x_b_recon.append(self.gen.decode(c_b, s_b_concat))

            # 使用分离出来的content code， 结合符合正太分布随机生成的style code，生成图片
            x_ba.append(self.gen.decode(c_b, s_a_concat.unsqueeze(0)))
            x_ab.append(self.gen.decode(c_a, s_b_concat.unsqueeze(0)))
            

        # 把图片的像素连接起来
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba