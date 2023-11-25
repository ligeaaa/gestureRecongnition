import numpy as np
import cv2

from PIL import Image

from cnn.data.utils import read_split_data

# img_h, img_w = 32, 32
img_h, img_w = 128, 128   #根据自己数据集适当调整，影响不大
means, stdevs = [], []
img_list = []

# 图片保存地址
imgs_path = 'C:/Users/离歌/Desktop/Data/train'

train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(imgs_path)

imgs_path_list = train_images_path

len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = Image.open(item)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # img = cv2.imread(os.path.join(imgs_path,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i,'/',len_)

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
