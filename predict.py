import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable
from torchvision import datasets



test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])
to_pil = transforms.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=torch.load('./cricket-model-5.pth',map_location={'cuda:0': 'cpu'})
# data_dir = './train'


# def get_random_images(num):
#     data = datasets.ImageFolder(data_dir, transform=test_transforms)
#     classes = data.classes
#     indices = list(range(len(data)))
#     np.random.shuffle(indices)
#     idx = indices[:num]
#     from torch.utils.data.sampler import SubsetRandomSampler
#     sampler = SubsetRandomSampler(idx)
#     loader = torch.utils.data.DataLoader(data, 
#                    sampler=sampler, batch_size=num)
#     dataiter = iter(loader)
#     images, labels = dataiter.next()
#     return images, labels, classes


def predict_image(image):
	image = to_pil(image)
	image_tensor = test_transforms(image).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input = Variable(image_tensor)
	output = model(input)
	#print(output)
	#print(output[0][0])
	#print(output[0][1])
	if(output[0][0] > -0.9 or output[0][1] < -2.5 ):
		index=0
	else:
		index=1
	#index = output.data.cpu().numpy().argmax()
	return index



#frame = cv2.imread('./74.jpg',cv2.IMREAD_UNCHANGED)
#print(predict_image(frame))





# images, labels, classes = get_random_images(5)
# fig=plt.figure(figsize=(10,10))
# for ii in range(len(images)):
#     image = to_pil(images[ii])
#     index = predict_image(image)
#     sub = fig.add_subplot(1, len(images), ii+1)
#     res = int(labels[ii]) == index
#     sub.set_title(str(classes[index]) + ":" + str(res))
#     plt.axis('off')
#     plt.imshow(image)
# plt.show()
# plt.savefig('result.png')




