number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHA_LOWER = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

img_height = 60  #图片高度 (此参数不可更改)
img_width = 160     #图片宽度 (此参数不可更改)
# (指定图片文件路径,程序自动处理)

max_captcha = 6   #验证码长度
char_len = len(ALPHA_LOWER+number)  #所有验证码类型

train_size = 1032 #训练集个数
test_size = 112  #测试集个数

image_train_path = r'C:\Users\Administrator\Desktop\验证码训练\train_resize'  #训练集路径
image_test_path = r'C:\Users\Administrator\Desktop\验证码训练\test_resize'  #测试集路径


acc_save = 0.9  #模型准确率达到0.9,训练完成
acc_step = 100  #每100步测试一次准确率

step_save = 1000  #每训练1000次保存一次模型
step_all = 160000 #训练步数大于120000保存模型，训练完成
step_now = 121000  #用做预测模型时加载模型填的参数，训练时可不填(如调用test.ckpt-100.index,则填100)

model_save_path = r"./model/first.ckpt"  #模型保存路径

model_path = r'./model/first.ckpt-100'  #调用模型路径(load_model使用)

load_last_train = True  #是否接着上次的步数继续训练



