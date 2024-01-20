from networks import AdaINGen, MsImageDis
from utils import weights_init, get_model_list, kd_model_preprocess, vgg_preprocess, load_vgg16_4_1, load_vgg16_5_1, get_scheduler, load_shapeNet
from load_kd_module import load_kd_model
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import torchvision.transforms as tf
from torchvision import transforms
import numpy as np

class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        lr2 = hyperparameters['lr_dis']
        # Initiate the networks
        self.gen = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b

        # 使用正则化的方式
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        # style 输出的特征码维度
        self.style_dim = hyperparameters['gen']['style_dim']

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr2, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization,网络模型权重初始化
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # attention module
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

    def get_modal_code(self):
        one = torch.ones(1, 50).cuda()
        zero = torch.zeros(1, 50).cuda()
        t1_onehot = torch.cat((one, zero), 1).cuda()    #[1,1,...,1,0,0,...,0]
        t2_onehot = torch.cat((zero, one), 1).cuda()    #[0,0,...,0,1,1,...,1]
        t1_noise = 0.02*torch.randn(1, 100).cuda()
        t2_noise = 0.02*torch.randn(1, 100).cuda()
        t1_modalcode = t1_onehot + t1_noise
        t2_modalcode = t2_onehot + t2_noise
        return t1_modalcode, t2_modalcode


    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))


    def calc_modal_loss(self, input, target):
        return torch.sum(torch.abs(input - target))

    def forward(self, x_a, x_b):

        t1_modalcode, t2_modalcode = self.get_modal_code()

        t1_modalcode = t1_modalcode.unsqueeze(2)
        m_a = t1_modalcode.unsqueeze(2)
        t2_modalcode = t2_modalcode.unsqueeze(2)
        m_b = t2_modalcode.unsqueeze(2)                 #[1,100,1,1]

        self.eval()
        # input image encode, get cotent and style code
        c_a, s_a_fake = self.gen.encode(x_a)
        c_b, s_b_fake = self.gen.encode(x_b)

        s_a_concat = torch.cat((s_a_fake, m_a), 1)
        s_b_concat = torch.cat((s_b_fake, m_b), 1)
        # print(s_a_concat.size())

        # b-content and a-style fusion
        x_ba = self.gen.decode(c_b, s_a_concat)

        x_ab = self.gen.decode(c_a, s_b_concat)
        self.train()
        return x_ab, x_ba

    # 生成模型进行优化
    def gen_update(self, x_a, x_b, hyperparameters):
    	# 给输入的 x_a, x_b 加入随机噪声
        self.gen_opt.zero_grad()

        t1_modalcode, t2_modalcode = self.get_modal_code()

        t1_modalcode = t1_modalcode.unsqueeze(2)
        m_a = t1_modalcode.unsqueeze(2)
        t2_modalcode = t2_modalcode.unsqueeze(2)
        m_b = t2_modalcode.unsqueeze(2) 

        # a, b 图像进行编码
        c_a, s_a_prime = self.gen.encode(x_a)
        c_b, s_b_prime = self.gen.encode(x_b)

        s_a_concat = torch.cat((s_a_prime, m_a), 1)
        s_b_concat = torch.cat((s_b_prime, m_b), 1)
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

        s_a_concat_recon = torch.cat((s_a_recon, m_a), 1)
        s_b_concat_recon = torch.cat((s_b_recon, m_b), 1)


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
        self.loss_gen_adv_ab = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_ba = self.dis_b.calc_gen_loss(x_ab)

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
    
    # 训练的时候，查看目前的效果
    def sample(self, x_a, x_b):
        self.eval()

        t1_modalcode, t2_modalcode = self.get_modal_code()

        t1_modalcode = t1_modalcode.unsqueeze(2)
        m_a = t1_modalcode.unsqueeze(2)
        t2_modalcode = t2_modalcode.unsqueeze(2)
        m_b = t2_modalcode.unsqueeze(2)

        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []

        for i in range(x_a.size(0)):
        	# 输入图片a,b，分别得到对应的content code 以及 style code
            c_a, s_a_fake = self.gen.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0))

            s_a_concat = torch.cat((s_a_fake, m_a), 1)
            s_b_concat = torch.cat((s_b_fake, m_b), 1)
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
    
    # 鉴别模型进行优化
    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()

        t1_modalcode, t2_modalcode = self.get_modal_code()

        t1_modalcode = t1_modalcode.unsqueeze(2)
        m_a = t1_modalcode.unsqueeze(2)
        # print()
        t2_modalcode = t2_modalcode.unsqueeze(2)
        m_b = t2_modalcode.unsqueeze(2) 

        c_a, s_a = self.gen.encode(x_a)
        c_b, s_b = self.gen.encode(x_b)

        s_a_concat = torch.cat((s_a, m_a), 1)
        # print(s_a_concat.size())
        s_b_concat = torch.cat((s_b, m_b), 1)

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
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)