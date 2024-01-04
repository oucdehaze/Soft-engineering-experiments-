# dehaze.py import
from pyexpat import model
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
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
import argparse
#Models lib
from models import *
#Metrics lib
from metrics import calc_psnr, calc_ssim
from runpy import run_path
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
import cv2
import pymysql

# flask修改尝试
app=Flask(__name__)

src = 'https://840y096t32.goho.co/static/image/2.jpg'
src1 = 'https://840y096t32.goho.co/static/image/1.jpg'
src2 = 0
rain='https://840y096t32.goho.co/static/image/2-2.jpg'
rain1='https://840y096t32.goho.co/static/image/1-2.jpg'
raining = 'https://840y096t32.goho.co/static/image/2-3.jpg'
raining1='https://840y096t32.goho.co/static/image/1-3.jpg'
flag=0

#根据需要修改这里的内容
@app.route('/')
def login():
   global src,src1,src2,rain,rain1,flag,raining,raining1
   flag=0
   return render_template("login.html")

@app.route('/register',methods=['GET', 'POST'])
def register():
  global src,src1,src2,rain,rain1,flag,raining,raining1
  if request.method == 'POST':
      username = request.form['username']
      password = request.form['password']
      print(username)
      print(password)
      db = pymysql.connect(host='localhost',
                           user='root',
                           password='7163865',
                           database='picture')
      cursor = db.cursor()
      cursor.execute("insert into Staffs values( '%s',  '%s')" % (username, password))
      db.commit()
      db.rollback()
      return render_template("login.html")

#根据需要修改这里的内容
# @app.route('/')
# def index():
@app.route('/image',methods=['GET', 'POST'])
def index():
  global src,src1,src2,rain,rain1,flag,raining,raining1
  #data={"src":src,"src1":src1,"rain":rain,"rain1":rain1,"raining":raining,"raining1":raining1,"src2":src2}
  if flag==0:
    username = request.form['username']
    password = request.form['password']
    print(username)
    print(password)
    # sql = 'SELECT password FROM Staffs WHERE usernpoame = %s'
    # helper = MysqlHelper(user='root',passwd='155955',db='picture')
    # result = helper.cha_all(sql,[username])
    db = pymysql.connect(host='localhost',
                   user='root',
                   password='7163865',
                   database='picture')
    cursor = db.cursor()
    cursor.execute("SELECT * FROM Staffs")
    while True:
      singleData = cursor.fetchone()
      if singleData is None:
        break
      print(singleData[0])
      print(singleData[1])
      if singleData[0]==username and singleData[1]==password and flag==0:
        print("true")
        flag=1
        db.rollback()
        return render_template("image.html",src='https://840y096t32.goho.co/static/image/2.jpg',src1='https://840y096t32.goho.co/static/image/1.jpg',src2 = 0,rain='https://840y096t32.goho.co/static/image/2-2.jpg',rain1='https://840y096t32.goho.co/static/image/1-2.jpg',raining='https://840y096t32.goho.co/static/image/2-3.jpg',raining1='https://840y096t32.goho.co/static/image/1-3.jpg')
    else:
      return render_template("error.html") 
  else:
     return render_template("image.html",src=src,src1=src1,src2 = src2,rain=rain,rain1=rain1,raining=raining,raining1=raining1)

    #在登陆设置初值，登陆时flag=1
    # global src,src1,src2,rain,rain1,flag,raining,raining1
    # if flag==0:
    #   flag=1
    #   return render_template("image.html",src='https://840y096t32.goho.co/static/image/2.jpg',src1='https://840y096t32.goho.co/static/image/1.jpg',src2 = 0,rain='https://840y096t32.goho.co/static/image/2-2.jpg',rain1='https://840y096t32.goho.co/static/image/1-2.jpg',raining='https://840y096t32.goho.co/static/image/2-3.jpg',raining1='https://840y096t32.goho.co/static/image/1-3.jpg')
    # return render_template("image.html",src=src,src1=src1,src2 = src2,rain=rain,rain1=rain1,raining=raining,raining1=raining1)

@app.route('/dehaze',methods=['POST'])
def dehaze_image():
  global src,src1,src2,rain,rain1,raining,raining1
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
  src='https://840y096t32.goho.co/static/image/test.jpg'
  src1='https://840y096t32.goho.co/static/image/put.jpg'
  src2=1
  data={"src":src,"src1":src1,"rain":rain,"rain1":rain1,"raining":raining,"raining1":raining1,"src2":1}
  return jsonify(data)

##### 还需要修改保存图片的逻辑
@app.route('/derain',methods=['POST'])
def derain():
  global src,src1,src2,rain,rain1,raining,raining1
  rain_path = request.files['file']

  model = Generator().cpu()
  model.load_state_dict(torch.load('./snapshot/gen.pkl', map_location=torch.device('cpu')))

  data_rain = Image.open(rain_path).convert('RGB')
  data_rain = (np.asarray(data_rain)/255.0)

  data_rain = torch.from_numpy(data_rain).float()
  data_rain = data_rain.permute(2,0,1)
  data_rain = data_rain.unsqueeze(0)

  out = model(data_rain)[-1]
  torchvision.utils.save_image(data_rain, "static/image/put1.jpg")
  torchvision.utils.save_image(out, "static/image/test1.jpg")
  # cv2.imwrite('static/image/test1.jpg', out)

  rain='https://840y096t32.goho.co/static/image/test1.jpg'
  rain1='https://840y096t32.goho.co/static/image/put1.jpg'
  src2=2
  data={"src":src,"src1":src1,"rain":rain,"rain1":rain1,"raining":raining,"raining1":raining1,"src2":2}
  return jsonify(data)

parser = argparse.ArgumentParser(description='Demo MPRNet')
# parser.add_argument('--task', required=True, type=str, help='Task to run', choices=['Deblurring', 'Denoising', 'Deraining'])

args = parser.parse_args()

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

@app.route('/deraining',methods=['POST'])
def deraining():
    global src,src1,src2,rain,rain1,flag,raining,raining1
    rain_path = request.files['file']

    load_file = run_path("MPRNet.py")
    model = load_file['MPRNet']()
    model.cpu()

    #model = Generator().cpu()
    weights = './snapshot/model_deraining.pth'
    load_checkpoint(model, weights)
    model.eval()
    #model.load_state_dict(torch.load('./snapshot/model_deraining.pth', map_location=torch.device('cpu')))

    data_rain = Image.open(rain_path).convert('RGB')
    data_rain = (np.asarray(data_rain)/255.0)

    data_rain = torch.from_numpy(data_rain).float()
    data_rain = data_rain.permute(2,0,1)
    data_rain = data_rain.unsqueeze(0)

    out = model(data_rain)[-1]    
    
    torchvision.utils.save_image(data_rain, "static/image/put2.jpg")
    torchvision.utils.save_image(out, "static/image/test2.jpg")

    raining='https://840y096t32.goho.co/static/image/test2.jpg'
    raining1='https://840y096t32.goho.co/static/image/put2.jpg'
    src2=3
    data={"src":src,"src1":src1,"rain":rain,"rain1":rain1,"raining":raining,"raining1":raining1,"src2":3}
    return jsonify(data)
  
if __name__=="__main__":
    #这里学习怎么自己的电脑变为服务器并发布自己的项目 参考文章：https://blog.csdn.net/u014252871/article/details/70569889?fromshare=blogdetail
    #不过目前不急，可以等功能实现好在本机上调试好以后部署，这样做可以降低成本，缺点是关机后就没办法使用服务器了
    #也要商量一下部署在谁的服务器上，也可以用相似的方法将应用部署在板子上
    #app.run(port=80,host="10.140.222.209",debug=True)
    app.run(port=5000)