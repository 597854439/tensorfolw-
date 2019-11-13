#coding:utf-8
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
                image = image_resize_and_convert(image_train_path+'\\'+ all_image[random_file]) #出现该错误 list index out of range 是因为数据集大小设置错误
                image = np.array(image)
                return name,image
            print("该验证码命名错误：{}".format(name))
    if type == 'test':
        all_image = os.listdir(image_test_path)
        while True:
            random_file = random.randint(0, test_size-1)
            base = os.path.basename(image_test_path + '\\' + all_image[random_file])
            name = os.path.splitext(base)[0]
            image = image_resize_and_convert(image_test_path + '\\' + all_image[random_file])
            image = np.array(image)
            return name, image

#我的标签就是图片名
#标签转成向量
def name2vec(name):
    vector = np.zeros(max_captcha*char_len)
    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(name):
        idx = i * char_len + char2pos(c)
        vector[idx] = 1
    return vector

def name2vec_myself(name):
    vector = np.zeros(max_captcha*char_len)
    labels = []
    for letter in name:
        if letter.isdigit():
            labels.append(int(letter))
        else:
            labels.append(ord(letter) - ord("a")+10)
    for i, c in enumerate(labels):
        idx = i * char_len + c
        vector[idx] = 1
    return vector

def get_next_batch(batch_size=16,type='train'):
    batch_x = np.zeros([batch_size, img_height*img_width])
    batch_y = np.zeros([batch_size, max_captcha*char_len])
    for i in range(batch_size):
        name, image = get_name_image(type)
        batch_x[i, :] = 1*(image.flatten())/225
        batch_y[i, :] = name2vec_myself(name)
    return batch_x, batch_y

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

def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1, max_captcha, char_len])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, max_captcha, char_len]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        if load_last_train == True:
            ckpt = tf.train.latest_checkpoint(os.path.dirname(model_save_path))
            print(ckpt)
            if ckpt:
                saver.restore(sess, ckpt)
                step = int(ckpt.split('-')[-1])
                print('加载上次的模型')
            else:
                print("无已存在模型")
        acc = 0
        while True:
            batch_x, batch_y = get_next_batch(64,type='train')
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print("当前步数：",step, "当前loss：",loss_,"当前准确率：",acc)
            # 每100 step计算一次准确率
            if step % acc_step == 0:
                batch_x_test, batch_y_test = get_next_batch(100,type='test')
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("当前测试准确率----------{}-----------".format(acc))
            if step % step_save == 0:
                if step != 0:
                    saver.save(sess, model_save_path, global_step=step)
                if acc > acc_save or step>step_all:
                    saver.save(sess, model_save_path, global_step=step)
                    break
            step += 1


train_crack_captcha_cnn()




