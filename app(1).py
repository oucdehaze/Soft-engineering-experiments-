# dehaze.py import
from pyexpat import model
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
import pymysql
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

# class MysqlHelper:
#     def __init__(self,user,passwd,db,host='localhost',port=3306,charset='utf8'):  #注意这里有默认值的变量一定要放在没有默认值变量的后面
#         self.host = host
#         self.port = port
#         self.user = user
#         self.passwd = passwd
#         self.db = db
#         self.charset = charset

#     def open(self):
#         self.conn=connect(host=self.host,port=self.port,db=self.db,
#                      user=self.user,passwd=self.passwd ,charset=self.charset)
#         self.cursor=self.conn.cursor()

#     def close(self):
#         self.cursor.close()
#         self.conn.close()

#     def cud(self,sql,params):  #增加、修改、删除
#         try:
#             self.open()

#             self.cursor.execute(sql,params)
#             self.conn.commit()
#             print('ok')
#             self.close()
#         except Exception as e:
#             print(e)

#     def cha_all(self,sql,params=()):  #查询获取多个值
#         try:
#             self.open()

#             self.cursor.execute(sql,params)
#             result = self.cursor.fetchall()

#             self.close()

#             return result
#         except Exception as e:
#             print(e.message)


# flask修改尝试
app=Flask(__name__)

src = '/static/image/2.png'
src1 = '/static/image/1.png'
rain='/static/image/2-2.jpg'
rain1='/static/image/1-2.jpg'


# obsClient = ObsClient(
#  access_key_id='USWVGG2I9WLFIO7PMFSX',#刚刚下载csv文件里面的Access Key Id
# secret_access_key='SrXpHa7y6AAANSyumAtuebyrVl37cNeO1Spnqoxm',#刚刚下载csv文件里面的Secret Access Ke
#    server='https://ouc-picture-repair.obs.cn-east-3.myhuaweicloud.com'#这里的访问域名就是我们在桶的基本信息那里记下的东西
#)

#根据需要修改这里的内容
@app.route('/')
def index():
   return render_template("login.html")
@app.route('/register',methods=['GET', 'POST'])
def register():
  if request.method == 'POST':
      username = request.form['username']
      password = request.form['password']
      print(username)
      print(password)
      db = pymysql.connect(host='localhost',
                           user='root',
                           password='155955',
                           database='picture')
      cursor = db.cursor()
      cursor.execute("insert into Staffs values( '%s',  '%s')" % (username, password))
      db.commit()
      db.rollback()
      return render_template("login.html")
   

@app.route('/image',methods=['GET', 'POST'])
def image():
   if request.method == 'POST':
      username = request.form['username']
      password = request.form['password']
      print(username)
      print(password)
      # sql = 'SELECT password FROM Staffs WHERE usernpoame = %s'
      # helper = MysqlHelper(user='root',passwd='155955',db='picture')
      # result = helper.cha_all(sql,[username])
      db = pymysql.connect(host='localhost',
                     user='root',
                     password='155955',
                     database='picture')
      cursor = db.cursor()
      cursor.execute("SELECT * FROM Staffs")
      while True:
        singleData = cursor.fetchone()
        if singleData is None:
          break
        print(singleData[0])
        print(singleData[1])
        if singleData[0]==username and singleData[1]==password:
          print("true")
          db.rollback()
          return render_template("image.html",src='/static/image/1.png',src1='/static/image/2.png',src2 = 0,rain='/static/image/1-2.jpg',rain1='/static/image/2-2.jpg')       
      return render_template("error.html")  
       

   #vkjhfghfxc.uhgidiu

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
  torchvision.utils.save_image(data_hazy, "/static/image/put.jpg")
  torchvision.utils.save_image(clean_image, "/static/image/test.jpg")
  basepath = os.path.dirname(__file__)
  src='/static/image/test.jpg'
  src1='/static/image/put.jpg'
  return render_template("image.html",src=src,src1=src1,src2 = 1,rain=rain,rain1=rain1)

@app.route('/derain',methods=['POST'])
def derain():

  rain_path = request.files['file']

  model = Generator().cpu()
  model.load_state_dict(torch.load('./snapshot/gen.pkl', map_location=torch.device('cpu')))

  data_rain = Image.open(rain_path).convert('RGB')
  data_rain = (np.asarray(data_rain)/255.0)

  data_rain = torch.from_numpy(data_rain).float()
  data_rain = data_rain.permute(2,0,1)
  data_rain = data_rain.unsqueeze(0)

  out = model(data_rain)[-1]
  torchvision.utils.save_image(data_rain, "/static/image/put1.jpg")
  torchvision.utils.save_image(out, "/static/image/test1.jpg")
  # cv2.imwrite('static/image/test1.jpg', out)

  rain='/static/image/test1.jpg'
  rain1='/static/image/put1.jpg'
  return render_template("image.html",src=src,src1=src1,src2 = 2,rain=rain,rain1=rain1)
  
if __name__=="__main__":
    #这里学习怎么自己的电脑变为服务器并发布自己的项目 参考文章：https://blog.csdn.net/u014252871/article/details/70569889?fromshare=blogdetail
    #不过目前不急，可以等功能实现好在本机上调试好以后部署，这样做可以降低成本，缺点是关机后就没办法使用服务器了
    #也要商量一下部署在谁的服务器上，也可以用相似的方法将应用部署在板子上
    #app.run(port=80,host="10.140.222.209",debug=True)
    app.run(port=7000)