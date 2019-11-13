import requests
import base64
import os,json
from PIL import Image
import time

#下载验证码
def download_Verification_Code():
    image_save_path = r'C:\Users\Administrator\Desktop\data' #图片保存路径
    image_size = 150  #下载图片数
    url = "http://www.qybz.org.cn/log/getCheckCode.do?a=1569572284643"
    n = 1
    while n<image_size:
        res = requests.get(url)
        with open(image_save_path+'\{}.jpg'.format(n),'wb')as f:
            f.write(res.content)
        n+=1
        print("第{}张图片下载成功".format(n))

# download_Verification_Code()
#调用打码平台标注验证码
def Annotation_Verification_Code():
    image_path = r'C:\Users\Administrator\Desktop\data'
    image_list = os.listdir(image_path)
    url = "http://api.ttshitu.com/base64"
    for image in image_list:
        if image.split('.')[0].isdigit():
            os.remove(image_path+'/'+image)

            # name = Image.open(image_path+'/'+image)
            # print(image)
            # with open(image_path+'/'+image, "rb") as f:
            #     base64_data = base64.b64encode(f.read())
            #     data = {
            #         'username':'',
            #         'password':'',
            #         'typeid':'3',
            #         'softid':'',
            #         'image':base64_data
            #     }
            #     res = requests.post(url,data=data)
            #     result = json.loads(res.text)
            #     code = result.get("data").get("result")
            #     outfile = image_path + '/' + code + '.jpg'
            #     name.save(outfile)
            #     print(image_path+'/'+image)
            #     print(outfile)
                # time.sleep(50000)

Annotation_Verification_Code()
