"""
细分类数据集制作
"""
import torch
import torch.utils.data as data
import torchvision.transforms as tf
import PIL.Image
import numpy as np


class ImageDataset(data.Dataset):
    def __init__(self, transform=None, settype=0,
                 rootfolder='/home/bobo/Downloads/dataset/png16',
                 nbchannel=1):
        # TODO
        # 1. Initialize file path or list of file names.
        self.rootFolder = rootfolder
        self.nbC = nbchannel
        self.pn = settype
        self.nrrd = [
                     '%s/trainlist1.txt' % self.rootFolder,
                     '%s/trainlist2.txt' % self.rootFolder,
                     '%s/trainlist3.txt' % self.rootFolder,
                     '%s/trainlist4.txt' % self.rootFolder,
                     '%s/trainlist5.txt' % self.rootFolder,
                     '%s/testlist1.txt' % self.rootFolder,
                     '%s/testlist2.txt' % self.rootFolder,
                     '%s/testlist3.txt' % self.rootFolder,
                     '%s/testlist4.txt' % self.rootFolder,
                     '%s/testlist5.txt' % self.rootFolder,
                     ]

        self.nrrd = self.nrrd[self.pn]

        self.nrrd_list1 = []            # label为0，健康
        self.nrrd_list2 = []            # label为1，HER2
        self.nrrd_list3 = []            # label为2，LA
        self.nrrd_list4 = []            # label为3，LB
        self.nrrd_list5 = []            # label为4，TNBC

        self.mask_list1 = []            # 与label同理
        self.mask_list2 = []
        self.mask_list3 = []
        self.mask_list4 = []
        self.mask_list5 = []

        self.label_list1 = []           # 与label同理
        self.label_list2 = []
        self.label_list3 = []
        self.label_list4 = []
        self.label_list5 = []

        file = open(self.nrrd, 'r', encoding='utf-8')
        line = file.readline()

        while line:
            line = line.strip('\n').strip()
            # print('line1:', line)
            if line[0] != '#':
                line = line.replace('./', '/')
                label = line[-1]
                # print(label)
                line = line[:-4]
                # print(line)
                fh = '%s%s' % (self.rootFolder, line)
                # print('fh:', fh)
                mh = '%s/%s%s' % (self.rootFolder, 'Mask', line[6:])
                mh = '%s00001.png' % (mh[0:-9])

                if fh is not None and label == '0':
                    # self.nrrd_id.append(temp_id)
                    self.nrrd_list1.append(fh)
                    self.mask_list1.append(mh)
                    self.label_list1.append(label)
                    # print(self.label_list)
                elif fh is not None and label == '1':
                    self.nrrd_list2.append(fh)
                    self.mask_list2.append(mh)
                    self.label_list2.append(label)
                elif fh is not None and label == '2':
                    self.nrrd_list3.append(fh)
                    self.mask_list3.append(mh)
                    self.label_list3.append(label)
                elif fh is not None and label == '3':
                    self.nrrd_list4.append(fh)
                    self.mask_list4.append(mh)
                    self.label_list4.append(label)
                elif fh is not None and label == '4':
                    self.nrrd_list5.append(fh)
                    self.mask_list5.append(mh)
                    self.label_list5.append(label)

            else:
                print('pass==> {}'.format(line))

            line = file.readline()
        # print(self.nrrd_list1)
        # print(self.nrrd_list2)
        file.close()
        # print('list1', self.label_list1)
        self.set_len = (len(self.nrrd_list1)) * 5  # 以没有病的样本作为数据集的长度

        # print('数据集长度:', self.set_len)
        # print('最长子词条：', len(self.nrrd_list1))

        self.preTransform = transform
        self.initperm()

    def initperm(self):
        self.nrrd_perm = torch.randperm(self.set_len)
        # print(self.nrrd_perm, len(self.nrrd_perm))

    def getindex(self, index):
        index = self.nrrd_perm[index % self.set_len]
        # print('index:', int(index))
        return index

    def loadimage(self, path1, path2):

        img = PIL.Image.open(path1)
        img.load()
        img = tf.Resize((400, 400))(img)

        msk = PIL.Image.open(path2)
        msk.load()
        msk = tf.Resize((400, 400))(msk)

        if self.preTransform:
            # NOTE, potential bug that different transform will be apply on image and mask coursed by random operations.

            baseimg = tf.ToTensor()(img)
            basemsk = tf.ToTensor()(msk)
            baseimg = baseimg.float() / 65535                   # norm to [0, 1]
            basemsk = basemsk.float() / 65535
            merge_img = torch.cat((baseimg, basemsk))           # merge img and mask
            merge_img = tf.ToPILImage()(merge_img.float())
            merge_img = self.preTransform(merge_img)
            img, msk = merge_img.chunk(2, 0)

        img = img.float()
        msk = msk.float()

        img_mean = img.mean()
        img_std = img.std()
        img = (img - img_mean) / img_std

        if self.nbC > 1:
            img = img.repeat(self.nbC, 1, 1)
            msk = msk.repeat(self.nbC, 1, 1)

        return img, msk

    def __getitem__(self, index):

        index = self.getindex(index)
        index = int(index)
        # print('最长子词条', len(self.nrrd_list1))
        if index // len(self.nrrd_list1) == 0:
            # print('0')
            # print('0:', len(self.nrrd_list1), (index % len(self.nrrd_list1)) % len(self.nrrd_list1))
            img, msk = self.loadimage(self.nrrd_list1[(index % len(self.nrrd_list1)) % len(self.nrrd_list1)],
                                      self.mask_list1[(index % len(self.mask_list1)) % len(self.nrrd_list1)],)

            img3d = img
            msk3d = msk

            for i in range(9):
                '在深度维度上进行复制'
                img3d = torch.cat([img3d, img], dim=0)
                msk3d = torch.cat([msk3d, msk], dim=0)

            pixels = img3d[img3d > 0]
            mean = pixels.mean()
            std = pixels.std()
            img3d = (img3d - mean) / std
            out_random = np.random.normal(0, 1, size=img3d.shape)
            out_random = torch.from_numpy(out_random).float()
            img3d[img3d == 0] = out_random[img3d == 0]

            img2d = img3d[4:5]

            label = self.label_list1[(index % len(self.nrrd_list1)) % len(self.nrrd_list1)]
            return img2d, msk, img3d, msk3d, label

        if index // len(self.nrrd_list1) == 1:
            # print('1')
            # print('1:', len(self.nrrd_list2), (index % len(self.nrrd_list1)) % len(self.nrrd_list2))
            img, msk = self.loadimage(self.nrrd_list2[(index % len(self.nrrd_list1)) % len(self.nrrd_list2)],
                                      self.mask_list2[(index % len(self.mask_list1)) % len(self.nrrd_list2)],)

            img3d = img
            msk3d = msk

            for i in range(9):
                '在深度维度上进行复制'
                img3d = torch.cat([img3d, img], dim=0)
                msk3d = torch.cat([msk3d, msk], dim=0)

            pixels = img3d[img3d > 0]
            mean = pixels.mean()
            std = pixels.std()
            img3d = (img3d - mean) / std
            out_random = np.random.normal(0, 1, size=img3d.shape)
            out_random = torch.from_numpy(out_random).float()
            img3d[img3d == 0] = out_random[img3d == 0]

            img2d = img3d[4:5]

            label = self.label_list2[(index % len(self.nrrd_list1)) % len(self.nrrd_list2)]

            return img2d, msk, img3d, msk3d, label

        if index // len(self.nrrd_list1) == 2:
            # print('2')
            # print('2:', len(self.nrrd_list3), (index % len(self.nrrd_list1)) % len(self.nrrd_list3))
            img, msk = self.loadimage(self.nrrd_list3[(index % len(self.nrrd_list1)) % len(self.nrrd_list3)],
                                      self.mask_list3[(index % len(self.mask_list1)) % len(self.nrrd_list3)],)

            img3d = img
            msk3d = msk

            for i in range(9):
                '在深度维度上进行复制'
                img3d = torch.cat([img3d, img], dim=0)
                msk3d = torch.cat([msk3d, msk], dim=0)

            pixels = img3d[img3d > 0]
            mean = pixels.mean()
            std = pixels.std()
            img3d = (img3d - mean) / std
            out_random = np.random.normal(0, 1, size=img3d.shape)
            out_random = torch.from_numpy(out_random).float()
            img3d[img3d == 0] = out_random[img3d == 0]

            img2d = img3d[4:5]

            label = self.label_list3[(index % len(self.nrrd_list1)) % len(self.nrrd_list3)]

            return img2d, msk, img3d, msk3d, label

        if index // len(self.nrrd_list1) == 3:
            # print('3')
            # print('3:', len(self.nrrd_list4), (index % len(self.nrrd_list1)) % len(self.nrrd_list4))
            img, msk = self.loadimage(self.nrrd_list4[(index % len(self.nrrd_list1)) % len(self.nrrd_list4)],
                                      self.mask_list4[(index % len(self.mask_list1)) % len(self.nrrd_list4)],)

            img3d = img
            msk3d = msk

            for i in range(9):
                '在深度维度上进行复制'
                img3d = torch.cat([img3d, img], dim=0)
                msk3d = torch.cat([msk3d, msk], dim=0)

            pixels = img3d[img3d > 0]
            mean = pixels.mean()
            std = pixels.std()
            img3d = (img3d - mean) / std
            out_random = np.random.normal(0, 1, size=img3d.shape)
            out_random = torch.from_numpy(out_random).float()
            img3d[img3d == 0] = out_random[img3d == 0]

            img2d = img3d[4:5]

            label = self.label_list4[(index % len(self.nrrd_list1)) % len(self.nrrd_list4)]

            return img2d, msk, img3d, msk3d, label

        if index // len(self.nrrd_list1) == 4:
            # print('4')
            # print('4:', len(self.nrrd_list5), (index % len(self.nrrd_list1)) % len(self.nrrd_list5))
            img, msk = self.loadimage(self.nrrd_list5[(index % len(self.nrrd_list1)) % len(self.nrrd_list5)],
                                      self.mask_list5[(index % len(self.mask_list1)) % len(self.nrrd_list5)],)

            img3d = img
            msk3d = msk

            for i in range(9):
                '在深度维度上进行复制'
                img3d = torch.cat([img3d, img], dim=0)
                msk3d = torch.cat([msk3d, msk], dim=0)

            label = self.label_list5[(index % len(self.nrrd_list1)) % len(self.nrrd_list5)]

            pixels = img3d[img3d > 0]
            mean = pixels.mean()
            std = pixels.std()
            img3d = (img3d - mean) / std
            out_random = np.random.normal(0, 1, size=img3d.shape)
            out_random = torch.from_numpy(out_random).float()
            img3d[img3d == 0] = out_random[img3d == 0]

            img2d = img3d[4:5]

            return img2d, msk, img3d, msk3d, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.set_len
