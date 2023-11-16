import torchvision
from PIL import Image
from cnnModel import *
import cv2
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 调用摄像头
cap = cv2.VideoCapture(0)

# 初始化参数
pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    if success:
        # 将numpy类型转化为Image类型
        image = Image.fromarray(img)
        # 转化其为模型的输入类型
        image = image.convert('RGB')
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                    torchvision.transforms.ToTensor()])

        image = transform(image)
        # 调用GPU
        image = image.to(device)
        # 调用已加载好的模型
        cnn = torch.load("cnn_0.pth")
        # 调用GPU
        cnn = cnn.to(device)

        image = torch.reshape(image, (1, 3, 32, 32))
        cnn.eval()
        with torch.no_grad():
            output = cnn(image)
        print(output)
        answer = output.argmax(1)

        # 打印帧率
        # 当前时间
        cTime = time.time()
        # 计算fps
        fps = 1 / (cTime - pTime)
        # 存储当前时间
        pTime = cTime
        # 将fps显示在img上
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        # 打印结果
        cv2.putText(img, str(int(answer)), (270, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        # 将捕捉到的image展现在Image框中
        cv2.imshow("Image", img)

    # 按键等待1毫秒，允许窗口保持打开状态
    cv2.waitKey(1)