import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import *


test_transforms = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
to_pil = transforms.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('./cricket-model-Alexnet-1.pth',map_location={'cuda:0': 'cpu'})


def image_loader(image):
    image = to_pil(image)
    image = test_transforms(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def Pilcropped(image):
    width, height  = image.size
    left = 0
    top = height*.8
    right = width
    bottom = height
    image = image.crop((left, top, right, bottom))
    return image


def predict_image(image):
	image.resize((500,281), Image.ANTIALIAS)
	image = Pilcropped(image)
	image_tensor = test_transforms(image).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input = Variable(image_tensor)
	output = model(input)
	index = output.data.cpu().numpy().argmax()
	return index


frame = Image.open('./1185.jpg')
Prediction = predict_image(frame)
print(Prediction)

