import cv2
import mediapipe as mp
import numpy as np
import time

import torch
import torchvision
from PIL import Image

# 打开默认摄像头
cap = cv2.VideoCapture(0)

# 创建一个手的实体类来存储手部信息
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# 创建一个用于绘制手部关键点和连接线的实体类
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while True:
    # 从摄像头捕获一帧图像
    success, img = cap.read()
    draw_img = np.copy(img)

    # 将捕获的图像从BGR颜色空间转换为RGB颜色空间
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 处理图像以检测手部信息
    results = hands.process(imgRGB)

    # 检查是否检测到了手部关键点
    if results.multi_hand_landmarks:

        # 初始化
        concatenated_img = None
        imgs = []

        # 遍历每个检测到的手
        for handLms in results.multi_hand_landmarks:

            # 寻找手的边界框
            x_list = [lm.x for lm in handLms.landmark]
            y_list = [lm.y for lm in handLms.landmark]
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            # 获取矩形的左上角坐标和边长宽度
            top_left_x, top_left_y = int(xmin * img.shape[1]), int(ymin * img.shape[0])
            length = int((xmax - xmin) * img.shape[1])
            width = int((ymax - ymin) * img.shape[0])

            # 裁剪矩形区域
            hand_img = img[max(top_left_y-15,0) : min(top_left_y + width + 30,img.shape[0]),
                       max(top_left_x-15,0) : min(top_left_x + length + 30,img.shape[1])]

            # 将宽度 resize 成 32
            if hand_img is not None:
                # 等比例缩放
                max_dim = 32
                if length > width:
                    new_length = max_dim
                    new_width = int(max_dim * (width / length))
                else:
                    new_width = max_dim
                    new_length = int(max_dim * (length / width))


                hand_img_resized = cv2.resize(hand_img, (max(1, new_length), max(1, new_width)),
                                                interpolation=cv2.INTER_AREA)


                # 创建一个128x128的黑色背景
                background = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)

                # 计算将调整后的图像放置在背景中的位置
                y_offset = (max_dim - new_width) // 2
                x_offset = (max_dim - new_length) // 2

                # 将调整后的图像放置在背景中
                background[y_offset:y_offset + new_width, x_offset:x_offset + new_length] = hand_img_resized

                # 将numpy类型转化为Image类型
                image = Image.fromarray(background)
                # 转化其为模型的输入类型
                image = image.convert('RGB')
                transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                            torchvision.transforms.ToTensor()])

                image = transform(image)
                # 调用GPU
                image = image.to(device)
                # 调用已加载好的模型
                cnn = torch.load("cnn_81.pth")
                # 调用GPU
                cnn = cnn.to(device)

                image = torch.reshape(image, (1, 3, 32, 32))
                cnn.eval()
                with torch.no_grad():
                    output = cnn(image)
                print(output)
                answer = output.argmax(1)

                # 打印结果
                cv2.putText(draw_img, str(int(answer)), (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)

            # 绘制矩形
            cv2.rectangle(draw_img, (top_left_x - 15, top_left_y - 15), (top_left_x + length + 30, top_left_y + width + 30), (0, 255, 0), 2)

            # 获得手部21点的id和其对应的坐标
            # for id, lm in enumerate(handLms.landmark):
            #     # print(id,lm)
            #     h, w, c = img.shape
            #     cx, cy = int(lm.x*w), int(lm.y*h)
            #     print(id, cx, cy)
            #     # if id == 0:
            #     #     cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)

            # 在图像上绘制手部关键点和连接线
            mpDraw.draw_landmarks(draw_img, handLms, mpHands.HAND_CONNECTIONS)











    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(draw_img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
            (255,0,255),3)

    # 在窗口中显示图像
    cv2.imshow("Image", draw_img)
    # if concatenated_img is not None:
    #     cv2.imshow("Concatenated Hands", concatenated_img)

    # 按键等待1毫秒，允许窗口保持打开状态
    cv2.waitKey(1)
