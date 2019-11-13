#coding:utf-8
#模型预测代码
import tensorflow as tf
import os
import random
import numpy as np
from trains_code.setting import *
from trains_code.image_handle import image_resize_and_convert

def get_name_image(type):
    if type == 'train':
        all_image = os.listdir(image_train_path)
        while True:
            random_file = random.randint(0, train_size-1)
            base = os.path.basename(image_train_path + '\\' + all_image[random_file])
            name = os.path.splitext(base)[0]
            if len(str(name))==max_captcha:
                image = image_resize_and_convert(image_train_path+'\\'+ all_image[random_file])
                image = np.array(image)
                return name,image
    if type == 'test':
        all_image = os.listdir(image_test_path)
        while True:
            random_file = random.randint(0, test_size-1)
            base = os.path.basename(image_test_path + '\\' + all_image[random_file])
            name = os.path.splitext(base)[0]
            image = image_resize_and_convert(image_test_path + '\\' + all_image[random_file])
            image = np.array(image)
            return name, image

#将向量装成名字，也就是输出结果，最后预测的时候用到
def vec2name(vec):
    name = []
    for i, c in enumerate(vec):
        char_idx = c % char_len
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('a')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        name.append(chr(char_code))
    return "".join(name)

#CNN
def crack_captcha_cnn(w_alpha=0.01,b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, img_height, img_width, 1])
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3,3, 1, 32]),name='w_c1')
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]),name='b_c1')
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    #24*74*32
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]),name='w_c2')
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]),name='b_c2')
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    #11*37*64
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3,3, 64, 64]),name='w_c3')
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]),name='b_c3')
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    #8*20*64
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 32 * 40, 1024]),name='w_d')
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]),name='b_d')
    dense = tf.reshape(conv3,[-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob,name='dense')

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, max_captcha * char_len]),name='w_out')
    b_out = tf.Variable(b_alpha * tf.random_normal([max_captcha * char_len]),name='b_out')
    out = tf.add(tf.matmul(dense, w_out), b_out,name='predict')
    return out

X = tf.placeholder(tf.float32,[None,img_width*img_height],name='input_X')
Y = tf.placeholder(tf.float32,[None,max_captcha*char_len],name='input_Y')
keep_prob = tf.placeholder(tf.float32,name='keep_prob')


def crack_captcha():
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,model_save_path+'-'+str(step_now))
        n = 1
        accuar = 0
        while n <= 1000:
            name, image = get_name_image(type='test')
            image = 1 * (image.flatten())/225
            predict = tf.argmax(tf.reshape(output, [-1, max_captcha, char_len]), 2)
            name_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
            vec = name_list[0].tolist()
            predict_text = vec2name(vec)
            print("正确: {}  预测: {}".format(name, predict_text))
            n += 1
            if name==predict_text:
                accuar+=1
            print("模型预测准确率---------------------------------{}------------------------：".format(accuar/n))


crack_captcha()







