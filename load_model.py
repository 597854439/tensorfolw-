# .meta文件保存了当前图结构
# .index文件保存了当前参数名
# .data文件保存了当前参数值
import tensorflow as tf
import numpy as np
import base64,os
from trains_code.setting import *
from trains_code.image_handle import image_resize_and_convert
from trains_code.setting import model_path

#将向量装成名字，也就是输出结果，最后预测的时候用到

class predict():
    def __init__(self):
        self.sess = tf.Session()
        self.new_saver = tf.train.import_meta_graph(model_path+'.meta')
        self.new_saver.restore(self.sess, model_path)
        graph = tf.get_default_graph()
        self.y = graph.get_operation_by_name('predict').outputs[0]
        # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
        self.input_x = graph.get_operation_by_name('input_X').outputs[0]
        self.keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

    def vec2name(self,vec):
        name = []
        for i, c in enumerate(vec):
            char_idx = c % char_len
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            name.append(chr(char_code))
        return "".join(name)

    def get_result(self,image_base64,type='base64'):
        if type == 'path':
            image = image_resize_and_convert(image_base64)
            image = np.array(image)
            image = 1 * (image.flatten())/225
        elif type == 'base64':
            image = base64.b64decode(image_base64)
            file = open('1.jpg', 'wb')
            file.write(image)
            file.close()
            image = image_resize_and_convert('1.jpg')
            image = np.array(image)
            image = 1 * (image.flatten()) / 225
        else:
            raise ValueError('无 文件路径 或者base64字符串 ')
        # 使用y进行预测
        predict = self.sess.run(self.y, feed_dict={self.input_x:[image], self.keep_prob: 1.0})
        result = np.argmax(np.reshape(predict,[-1,max_captcha,char_len]),2)
        result = self.vec2name(result[0])
        print(result)
        return result


