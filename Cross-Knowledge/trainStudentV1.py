from torch import nn
import torch
import torchvision.transforms as tf
import ImageCopyDatasetV2 as Imagedataset_3D
from torch.utils.data import DataLoader
from setting import parse_opts
from model import generate_model
import progressbar
import os
import argparse
import module1
import module2
import module3

"""
细分类student网络训练
"""

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def train(teacher_model, student_model1, student_model2, dataloader, criterion, paramList):
    train_loss1_list = []
    train_loss2_list = []
    teacher_model.eval()
    student_model1.train()
    student_model2.train()
    student_model1.bn1.eval()
    number_of_entry = len(dataloader)

    loss_sum = 0.0
    loss2_sum = 0.0
    teacher_zero = torch.zeros(2, 2048, 50, 50)
    teacher_zero = teacher_zero.cuda()
    with progressbar.ProgressBar(max_value=number_of_entry) as bar:
        for i, item in enumerate(dataloader):
            img, msk, img3d, msk3d, label = item
            # print(img3d.size())
            if img3d.size() != (2, 10, 400, 400):
                teacher_zero = torch.zeros(1, 2048, 50, 50)
                teacher_zero = teacher_zero.cuda()
            img = img.cuda()
            img3d = img3d.unsqueeze(1)
            img3d = img3d.cuda()
            paramList['model1'].zero_grad()
            paramList['model2'].zero_grad()

            # 求teacher网络的第一层输出
            teacher_output = teacher_model.module.conv1(img3d)
            teacher_output = teacher_model.module.bn1(teacher_output)
            teacher_output = teacher_model.module.relu(teacher_output)
            teacher_output = teacher_model.module.maxpool(teacher_output)

            # 求teacher网络layer4的输出
            teacher_layer4_output_3d = teacher_model.module.layer1(teacher_output)
            teacher_layer4_output_3d = teacher_model.module.layer2(teacher_layer4_output_3d)
            teacher_layer4_output_3d = teacher_model.module.layer3(teacher_layer4_output_3d)
            teacher_layer4_output_3d = teacher_model.module.layer4(teacher_layer4_output_3d)

            teacher_layer4_output = torch.mean(teacher_layer4_output_3d, 2)

            # 求student网络倒数第二层的输出
            student_output = student_model1.conv1(img)
            student_output = student_model1.bn1(student_output)
            student_output = student_model1.relu1(student_output)
            student_output = student_model1.maxpool1(student_output)

            student_output = student_model2.layer1(student_output)
            student_output = student_model2.layer2(student_output)
            student_output = student_model2.layer3(student_output)
            student_layer4_output = student_model2.layer4(student_output)

            # loss反向传播
            loss = criterion(student_layer4_output, teacher_layer4_output)
            loss2 = criterion(teacher_zero, teacher_layer4_output)
            loss.backward()

            # paramList['model1'].step()
            paramList['model2'].step()

            loss_sum += float(loss.item())
            loss2_sum += float(loss2.item())
            bar.update(i)

    epoch_loss1 = round(float(loss_sum / number_of_entry), 3)
    epoch_loss2 = round(float(loss2_sum / number_of_entry), 3)

    loss_p = float(epoch_loss1 / epoch_loss2)
    train_loss1_list.append(epoch_loss1)
    train_loss2_list.append(loss_p)

    print('trainLoss1:', epoch_loss1, ' ', 'trainLoss2:', loss_p)
    return train_loss1_list, train_loss2_list


def verification(teacher_model, student_model1, student_model2, dataloader, criterion):
    verification_loss1_list = []
    verification_loss2_list = []
    teacher_model.eval()
    student_model1.eval()
    student_model2.eval()
    number_of_entry = len(dataloader)
    # scheduler.step()
    loss_sum = 0.0
    loss2_sum = 0.0
    teacher_zero = torch.zeros(2, 2048, 50, 50)
    teacher_zero = teacher_zero.cuda()
    with progressbar.ProgressBar(max_value=number_of_entry) as bar:
        with torch.no_grad():
            for i, item in enumerate(dataloader):
                img, msk, img3d, msk3d, label = item
                if img3d.size() != (2, 10, 400, 400):
                    teacher_zero = torch.zeros(1, 2048, 50, 50)
                    teacher_zero = teacher_zero.cuda()
                img = img.cuda()
                # print(img1.size())
                img3d = img3d.unsqueeze(1)

                img3d = img3d.cuda()

                # 求teacher网络的第一层输出
                teacher_output = teacher_model.module.conv1(img3d)
                teacher_output = teacher_model.module.bn1(teacher_output)
                teacher_output = teacher_model.module.relu(teacher_output)
                teacher_output = teacher_model.module.maxpool(teacher_output)

                # 求teacher网络layer4的输出
                teacher_layer4_output_3d = teacher_model.module.layer1(teacher_output)
                teacher_layer4_output_3d = teacher_model.module.layer2(teacher_layer4_output_3d)
                teacher_layer4_output_3d = teacher_model.module.layer3(teacher_layer4_output_3d)
                teacher_layer4_output_3d = teacher_model.module.layer4(teacher_layer4_output_3d)

                teacher_layer4_output = torch.mean(teacher_layer4_output_3d, 2)

                # 求student网络倒数第二层的输出
                student_output = student_model1.conv1(img)
                student_output = student_model1.bn1(student_output)
                student_output = student_model1.relu1(student_output)
                student_output = student_model1.maxpool1(student_output)

                student_output = student_model2.layer1(student_output)
                student_output = student_model2.layer2(student_output)
                student_output = student_model2.layer3(student_output)
                student_layer4_output = student_model2.layer4(student_output)

                # loss反向传播
                loss = criterion(student_layer4_output, teacher_layer4_output)
                loss2 = criterion(teacher_zero, teacher_layer4_output)
                loss_sum += float(loss.item())
                loss2_sum += float(loss2.item())
                bar.update(i)

    epoch_loss1 = round(float(loss_sum / number_of_entry), 3)
    epoch_loss2 = round(float(loss2_sum / number_of_entry), 3)

    loss_p = round(float(epoch_loss1 / epoch_loss2), 3)
    verification_loss1_list.append(epoch_loss1)
    verification_loss2_list.append(loss_p)

    print('verificationloss1:', epoch_loss1, ' ', 'verificationloss2:', loss_p)
    print(' ')
    return verification_loss1_list, verification_loss2_list


def save_model1(model, epoch, save_path):
    model_save_path = '{}epoch_{}.pth.tar'.format(save_path, epoch)
    model_save_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # print('Save checkpoints: epoch = {}'.format(epoch))
    torch.save({
        'ecpoch': epoch,
        'state_dict': model.state_dict(),
        'optimeizer': paramList['model1'].state_dict()},
        model_save_path)


def save_model2(model, epoch, save_path):
    model_save_path = '{}epoch_{}.pth.tar'.format(save_path, epoch)
    model_save_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # print('Save checkpoints: epoch = {}'.format(epoch))
    torch.save({
        'ecpoch': epoch,
        'state_dict': model.state_dict(),
        'optimeizer': paramList['model2'].state_dict()},
        model_save_path)


def save_log(filename, count, list1, list2, list3, list4):
    file = open(filename, 'a')
    for i in range(len(list1)):
        count = str(count)
        log = 'epoch' + count + ':  ' + 'trainloss1:' + str(list1[i]) + '   ' + 'trainloss2:' + str(list2[i]) + \
              '   ' + 'verificationloss1:' + str(list3[i]) + '   ' + 'verificationloss2:' + str(list4[i]) + '\n'
        file.write(log)
    file.close()


def parameter_replace(modelt, models):

    modelt_dict = modelt.state_dict()
    # for name, param in modelt.named_parameters():
    #     print(name, '      ', param.size())
    module_conv1_weight = modelt_dict['module.conv1.weight'].clone()
    module_bn1_weight = modelt_dict['module.bn1.weight'].clone()
    module_bn1_bias = modelt_dict['module.bn1.bias'].clone()
    # print(module_bn1_weight)
    # print(module_bn1_bias)
    module_conv1_weight_2d = torch.mean(module_conv1_weight, 2)
    # print(module_conv1_weight_2d.mean())
    module_bn1_weight_2d = module_bn1_weight
    module_bn1_bias_2d = module_bn1_bias
    # print(module_bn1_bias_2d.mean())
    models_dict = models.state_dict()
    # for name, param in models.named_parameters():
    #     print(name, '      ', param.size())
    models_dict['conv1.weight'].copy_(module_conv1_weight_2d)
    # print(models_dict['conv1.weight'].mean())
    # print(models_dict['conv1.weight'])
    models_dict['bn1.weight'].copy_(module_bn1_weight_2d)
    # print(models_dict['bn1.weight'])
    models_dict['bn1.bias'].copy_(module_bn1_bias_2d)
    # print(models_dict['bn1.bias'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--module1_resume_path',
        default='/home/work/bobo/seafile/Med/trails/细分类/modelStudent1/module1/epoch_9.pth.tar',
        type=str,
        help=
        'Path for module1 resume model.'
    )
    parser.add_argument(
        '--module2_resume_path',
        default='/home/work/bobo/seafile/Med/trails/细分类/modelStudent/module2/epoch_9.pth.tar',
        type=str,
        help=
        'Path for module2 resume model.'
    )
    args = parser.parse_args()

    '''
    teacher网络的构建
    '''
    sets_teacher = parse_opts()
    net_teacher, _ = generate_model(sets_teacher)
    '''
    学生网络分模块构建
    '''
    model1 = module1.generate_model1()
    model1 = model1.cuda()
    model2 = module2.generate_model2()
    model2 = model2.cuda()

    # '''
    # 模型读入参数
    # '''
    # checkpoint1 = torch.load(args.module1_resume_path)
    # # print('resume_path:', args.module1_resume_path, '\n')
    # model1.load_state_dict(checkpoint1['state_dict'])
    #
    # checkpoint2 = torch.load(args.module2_resume_path)
    # # print('resume_path:', args.module1_resume_path, '\n')
    # model2.load_state_dict(checkpoint2['state_dict'])

    '''
    将一个模块参数平均化复制
    '''
    parameter_replace(net_teacher, model1)

    '''
    训练网络参数配置
    '''
    pretransform = tf.Compose([
        tf.RandomVerticalFlip(),
        tf.RandomHorizontalFlip(),
        tf.ToTensor()
    ])

    trainset_3D = Imagedataset_3D.ImageDataset(transform=pretransform, settype=0,
                                               rootfolder='/home/work/bobo/png16',
                                               nbchannel=1)

    train_loader_3D = torch.utils.data.DataLoader(trainset_3D, batch_size=2, shuffle=True, num_workers=8)

    verificationset_3D = Imagedataset_3D.ImageDataset(transform=pretransform, settype=5,
                                                      rootfolder='/home/work/bobo/png16',
                                                      nbchannel=1)

    verificationset_loader_3D = torch.utils.data.DataLoader(verificationset_3D, batch_size=2, shuffle=True, num_workers=8)

    criterion = nn.MSELoss()

    # optimizer = torch.optim.SGD(model1.parameters(), lr=1e-3, momentum=0.9)

    lr1 = 1e-3
    lr2 = 1e-3

    model1_opt = torch.optim.SGD(list(model1.parameters())[0:1], lr=lr1, momentum=0.9, weight_decay=lr1/10)  # bypass bn parameters
    model2_opt = torch.optim.SGD(model2.parameters(), lr=lr2, momentum=0.9, weight_decay=lr2/10)
    paramList = {'model1': model1_opt, 'model2': model2_opt}
    # print(list(model1.parameters()))
    # print(list(model1.parameters())[0:1])

    for epoch in range(10):
        # print(list(model1.parameters()))
        print('Epoch {}:'.format(epoch))
        trainlist1, trainlist2 = train(net_teacher, model1, model2, train_loader_3D, criterion, paramList)
        verificationlist1, verificationlist2 = verification(net_teacher, model1, model2, verificationset_loader_3D, criterion)
        save_model1(model1, epoch, save_path="../trails/细分类/modelStudent1/module1/")
        save_model2(model2, epoch, save_path="../trails/细分类/modelStudent1/module2/")
        save_log('../trails/细分类/modelStudent1/log.txt', epoch, trainlist1, trainlist2, verificationlist1, verificationlist2)

