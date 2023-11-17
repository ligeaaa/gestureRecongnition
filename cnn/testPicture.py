import torchvision
from PIL import Image
from cnnModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "D:/code&project/gestureRecongnition/data/0.jpg"
image = Image.open(image_path)
image = image.convert('RGB')

print(type(image))

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
image = image.to(device)

pan = torch.load("cnn_8.pth")

image = torch.reshape(image, (1, 3, 32, 32))
pan.eval()
with torch.no_grad():
    output = pan(image)
print(output.argmax(1))