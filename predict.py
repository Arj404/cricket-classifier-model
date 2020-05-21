import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable



test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])
to_pil = transforms.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=torch.load('./cricket-model-3.pth',map_location={'cuda:0': 'cpu'})



def predict_image(image):
	image = to_pil(image)
	image_tensor = test_transforms(image).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input = Variable(image_tensor)
	output = model(input)
	index = output.data.cpu().numpy().argmax()
	return index


