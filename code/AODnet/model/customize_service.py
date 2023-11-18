from PIL import Image
import log
from model_service.pytorch_model_service import PTServingBaseService
import torch.nn.functional as F

import torch.nn as nn
import torch
import json

import numpy as np

logger = log.getLogger(__name__)

import torchvision.transforms as transforms

# 定义模型预处理
infer_transformation = transforms.Compose([
    transforms.Resize((28,28)),
    # 需要处理成pytorch tensor
    transforms.ToTensor()
])


import os


class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        # 调用父类构造方法
        super(PTVisionService, self).__init__(model_name, model_path)
        # 调用自定义函数加载模型
        self.model = Mnist(model_path)
        # # 加载标签
        # self.label = [0,1,2,3,4,5,6,7,8,9]
        # 亦可通过文件标签文件加载
        # model目录下放置label.json文件，此处读取
        # dir_path = os.path.dirname(os.path.realpath(self.model_path))
        # with open(os.path.join(dir_path, 'label.json')) as f:
        #     self.label = json.load(f)


    def _preprocess(self, data):
        # dehaze_net = dehaze_net().cuda()
        # dehaze_net.load_state_dict(torch.load("dehazer.pth"))

        preprocessed_data = {}
        for k, v in data.items():
            input_batch = []
            for file_name, file_content in v.items():
                with Image.open(file_content) as image1:
                    # 去雾处理
                    image1 = dehaze_net(image1);
                    if torch.cuda.is_available():
                        input_batch.append(infer_transformation(image1).cuda())
                    else:
                        input_batch.append(infer_transformation(image1))
            input_batch_var = torch.autograd.Variable(torch.stack(input_batch, dim=0), volatile=True)
            print(input_batch_var.shape)
            preprocessed_data[k] = input_batch_var

        return preprocessed_data

    def _postprocess(self, data):
        results = []
        for k, v in data.items():
            result = torch.argmax(v[0])
            result = {k: self.label[result]}
            results.append(result)
        return results

    def _inference(self, data):

        result = {}
        for k, v in data.items():
            result[k] = self.model(v)

        return result

class dehaze_net(nn.Module):

	def __init__(self):
		super(dehaze_net, self).__init__()

		self.relu = nn.ReLU(inplace=True)
	
		self.e_conv1 = nn.Conv2d(3,3,1,1,0,bias=True) 
		self.e_conv2 = nn.Conv2d(3,3,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(6,3,5,1,2,bias=True) 
		self.e_conv4 = nn.Conv2d(6,3,7,1,3,bias=True) 
		self.e_conv5 = nn.Conv2d(12,3,3,1,1,bias=True) 
		
	def forward(self, x):
		source = []
		source.append(x)

		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))

		concat1 = torch.cat((x1,x2), 1)
		x3 = self.relu(self.e_conv3(concat1))

		concat2 = torch.cat((x2, x3), 1)
		x4 = self.relu(self.e_conv4(concat2))

		concat3 = torch.cat((x1,x2,x3,x4),1)
		x5 = self.relu(self.e_conv5(concat3))

		clean_image = self.relu((x5 * x) - x5 + 1) 
		
		return clean_image



def Mnist(model_path, **kwargs):
    # 生成网络
    model = dehaze_net()
    # 加载模型
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(torch.load("/home/mind/model/dehazer.pth", map_location="cuda:0"))
    else:
        # device = torch.device('cpu')
        model.load_state_dict(torch.load("/home/mind/model/dehazer.pth", map_location='cpu'))
    # CPU或者GPU映射
    # model.to(device)
    # 声明为推理模式
    model.eval()

    return model
