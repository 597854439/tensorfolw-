from .setting import img_height,img_width
from PIL import Image

def image_resize_and_convert(file_path):  #图片重置大小与二值化处理
    im = Image.open(file_path)
    image = im.convert('L')
    out = image.resize((img_width, img_height), Image.ANTIALIAS)  # resize image with high-quality
    return out





