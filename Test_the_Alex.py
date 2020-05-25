import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable
from torchvision import datasets


test_transforms = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
to_pil = transforms.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=torch.load('./cricket-model-Alexnet-1.pth',map_location={'cuda:0': 'cpu'})
data_dir = './trainc'


def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, 
                   sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels, classes


def predict_image(image):
	image_tensor = test_transforms(image).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input = Variable(image_tensor)
	output = model(input)
	index = output.data.cpu().numpy().argmax()
	return index

lp,ln,rp,rn = 0,0,0,0
images, labels, classes = get_random_images(1000)
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    res = int(labels[ii]) == index
    if res == True:
        if index == 0:
            lp=lp+1
        if index == 1:
            rp=rp+1
    else:
        if index == 0:
            ln=ln+1
        if index == 1:
            rn=rn+1


print(lp,ln,rp,rn)
print(f'Accuracy = {((lp+rp)/(lp+rp+ln+rn))*100}')
print(f'False positive = {((ln)/(lp+rp+ln+rn))*100}')



