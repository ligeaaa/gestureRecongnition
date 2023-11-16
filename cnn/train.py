from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
import cnnModel

from cnn.data import readData

# 准备数据集

train_dataloader, test_dataloader = readData.getDataLader()
train_data_size = len(train_dataloader)
test_data_size = len(test_dataloader)
print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

# 创建网络模型
pan = cnnModel.Pan()
pan = pan.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(pan.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 设置训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 100

# 添加tensorboard
writer = SummaryWriter("../logs_train")


for i in range(epoch):
    print("----------第 {} 轮训练开始----------".format(i+1))
    # 训练步骤开始
    pan.train()  # 注意，这一步表示开启模型的训练模式，但是只有某些特定的层需要开启（详情查看官网），在本代码中仅仅是为了规范写上这行代码
    for step, data in enumerate(train_dataloader):
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = pan(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    pan.eval()  # 注意，这一步表示开启模型的测试模式，但是只有某些特定的层需要开启（详情查看官网），在本代码中仅仅是为了规范写上这行代码
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = pan(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # 保存每一轮的网络模型
    torch.save(pan, "pan_{}.pth".format(i))
    print("模型已保存")

writer.close()
