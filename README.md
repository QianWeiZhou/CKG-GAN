# Source code (Pytorch) for the paper: “Cross-Dimensional Knowledge-Guided Synthesizer Trained With Unpaired Multimodality MRIs”
# Contact: zhouqianweischolar@gmail.com (Qianwei Zhou) and 202103150302@zjut.edu.cn (Binjia Zhou)
# Configure the Enviroment
## Depending on speed of your Internet connection，installation may take hours.
## In our implemenation, we used python 3.8.18, Ubuntu 18.04.6 LTS with GPU.  
1. nvidia gpu driver version: 530.30.02
2. cuda version: 12.1
3. GPU memory >= 12GB
4. install miniconda
5. `$ conda create --name testENV --file package-list.txt -c pytorch`
6. `$ pip install pypng`

# You can train/test the CKG-GAN models by following instructions below.
1. BraTs2018: https://www.med.upenn.edu/sbia/brats2018/data.html
2. BraTs2021: http://braintumorsegmentation.org/
3. IXI dataset: https://brain-development.org/ixi-dataset/

# Prepare Imaging Data
1. Adjust original images to the resolutions reported in the paper.
2. Place traing data to folder /datasets/,
    1. In /datasets/BraTs2018/, list all images of BraTs2018.
    2. In /datasets/BraTs2021/, list all images of BraTs2021.
    3. In /datasets/IXI/, list all images of IXI dataset.

# Train CKG-GAN 
You can obtain the pre-trained segmentation model and the student model for link https://pan.baidu.com/s/1VAwsJxAg0WmQyobuwQFOUQ, the password is bfgq, then put them in the folder ./model.

`$ python brats_4type_train.py` to train image generator.
* Output:
    * The generator and discriminator models will be in the folder /outputs/brats_4type_train/checkpoints/.
    * They may look like dis_00040000.pt (the model of discriminator), gen_00040000.pt (the model of generator).

# Test the CKG-GAN on Real Different Type images
* In the file brats_4type_test.py, please set pre-train model of your own that you are going to test.  
* Please copy the target models (for example, gen_00040000.pt) to folder /outputs/brats_4type_train/checkpoints/.  

`$ python brats_4type_test.py`  

* Output: the code will generate target type images from input images.   
    * Generated fake images will be in the folder ./test
    * For example: /Samples/realImages/3917L-CC-neg.png ---> ./test/output_num0001.jpg

# The Work is Licensed with Apache License Version 2.0.
