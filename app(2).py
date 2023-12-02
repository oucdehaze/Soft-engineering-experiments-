# dehaze.py import
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
# from flask import  Flask,render_template
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import timedelta
# from obs import ObsClient

# flask修改尝试
app=Flask(__name__)

# obsClient = ObsClient(
#  access_key_id='USWVGG2I9WLFIO7PMFSX',#刚刚下载csv文件里面的Access Key Id
# secret_access_key='SrXpHa7y6AAANSyumAtuebyrVl37cNeO1Spnqoxm',#刚刚下载csv文件里面的Secret Access Key
#    server='https://ouc-picture-repair.obs.cn-east-3.myhuaweicloud.com'#这里的访问域名就是我们在桶的基本信息那里记下的东西
#)

#根据需要修改这里的内容
@app.route('/')
def index():
   return render_template("image.html",src='/static/image/2.png',src1='/static/image/1.png',src2 = 0)

@app.route('/dehaze',methods=['POST'])
def dehaze_image():

  #data_haze=request.form['data_haze']
  #这里接收前端传来的文件路径
  #需要思考上传的代码逻辑 参考文件：http://t.csdnimg.cn/L2WaR
  #上述内容在前端实现
  #如果有需要的话这里可以添加判断请求类型的if语句
  image_path = request.files['file']
  # print('路径%s' % image_path)

  data_hazy = Image.open(image_path).convert('RGB')
  data_hazy = (np.asarray(data_hazy)/255.0)

  data_hazy = torch.from_numpy(data_hazy).float()
  data_hazy = data_hazy.permute(2,0,1)
  data_hazy = data_hazy.unsqueeze(0)

  dehaze_net = net.dehaze_net()
  dehaze_net.load_state_dict(torch.load('snapshot/dehazer.pth'))

  clean_image = dehaze_net(data_hazy)
  torchvision.utils.save_image(data_hazy, "static/image/put.jpg")
  torchvision.utils.save_image(clean_image, "static/image/test.jpg")
  # torchvision.utils.save_image(clean_image, "test/" + image_path)
  # 下面的保存工作中的路径可以是云库的路径
  # 当前文件所在路径
  basepath = os.path.dirname(__file__)
  # 一定要先创建该文件夹，不然会提示没有该路径 basepath+''中的内容就是保存路径
  # MySQL需要添加连接数据库的语句，而且只有本地可以使用,也可以买和服务器配套的云空间，不需要连接数据库（但是服务器可以有免费的）
  # 也可以连接OBS库（已经购买过了）参考文章　http://t.csdnimg.cn/LDhfb
  # 可以将最后的secure_filename(dehaze_image.filename)修改为固定的文件名，这样可以节省空间，但是要考虑用户同时使用时的冲突问题，可以考虑登陆界面
  # 用OBS桶需要每隔一段时间清理一次
  # upload_path = os.path.join(basepath, 'stastic/images', secure_filename(dehaze_image.filename))
  # upload_path = 'D:\ruanjiangongcheng-homework\PyTorch-Image-Dehazing-master\code\flask\image.jpg'
  # resp = obsClient.putFile('ouc-picture-repair', 'images/', file_path=file)
  # clean_image.save(upload_path)
  
  #这里应该可以仿照java的上传文件写法上传图片并显示
  #方法一
  #session['clean_image']=cupload_path #使用session存储方式，session默认为数组，给定key和value即可
  #return render_template("dehaze.html")
  #方法二
  #需要在前端添加<img src="{{image}}" width="500" height="600">
  #return render_template("dehaze.html",mimetype="image/jpeg",image=upload_path)
  #使用OBS桶的话可能需要前端使用OBS临时url 参考文件：https://support.huaweicloud.com/perms-cfg-obs/obs_40_0009.html
  # return render_template("dehaze.html",mimetype="image/jpeg",image=secure_filename(dehaze_image.filename))
  return render_template("image.html",src='/static/image/test.jpg',src1='/static/image/put.jpg',src2 = 1)
  
if __name__=="__main__":
    #这里学习怎么自己的电脑变为服务器并发布自己的项目 参考文章：https://blog.csdn.net/u014252871/article/details/70569889?fromshare=blogdetail
    #不过目前不急，可以等功能实现好在本机上调试好以后部署，这样做可以降低成本，缺点是关机后就没办法使用服务器了
    #也要商量一下部署在谁的服务器上，也可以用相似的方法将应用部署在板子上
    #app.run(port=5000,host="10.1.12.47",debug=True)
    app.run()
