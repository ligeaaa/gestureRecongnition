import os
from multiprocessing import freeze_support

import torch
from torchvision import transforms

from cnn.data.my_dataset import MyDataSet
from cnn.data.utils import read_split_data, plot_data_loader_image

# root = "C:/Users/离歌/Desktop/shp_marcel_train/Marcel-Train"  # 数据集所在根目录
# root = "C:/Users/离歌/Desktop/Indian Sign Language Dataset_data_datasets"  # 数据集所在根目录
# root = "D:/桌面/shp_marcel_train/Marcel-Train"  # 数据集所在根目录
root = "C:/Users/离歌/Desktop/Data/train"  # 数据集所在根目录


def getDataLader():
    # 调用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(32),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(32),
                                   transforms.CenterCrop(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    val_data_set = MyDataSet(images_path=val_images_path,
                               images_class=val_images_label,
                               transform=data_transform["val"])

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=val_data_set.collate_fn)

    plot_data_loader_image(train_loader)

    # for step, data in enumerate(train_loader):
    #     print("123")
    #     images, labels = data

    return train_loader, val_loader

if __name__ == '__main__':
    freeze_support()
    getDataLader()