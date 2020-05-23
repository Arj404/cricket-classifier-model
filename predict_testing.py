import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable
from torchvision import datasets



test_transforms = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), ])
to_pil = transforms.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=torch.load('./cricket-model-6.pth',map_location={'cuda:0': 'cpu'})
data_dir = './trainc'


# def cropped(frame):
# 	height, width = frame.shape[:2]
# 	start_row, start_col = int(height * .8), int(width * 0)
# 	end_row, end_col = int(height*1), int(width*1)
# 	cropped = frame[start_row:end_row , start_col:end_col]
# 	#cropped2 = frame[start_row:end_row , start_col:end_col,1]
# 	#cropped3 = frame[start_row:end_row , start_col:end_col,2]
# 	#cropped = 
# 	return cropped


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
	#image = to_pil(image)
	image_tensor = test_transforms(image).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input = Variable(image_tensor)
	output = model(input)
	print(output)
	#print(output[0][0])
	#print(output[0][1])
	# if(output[0][0] > -0.9 or output[0][1] < -2.5 ):
	# 	index=0
	# else:
	# 	index=1
	index = output.data.cpu().numpy().argmax()
	print(index)
	return index



# frame = cv2.imread('./85.jpg',cv2.IMREAD_UNCHANGED)
# image = cropped(frame)
# plt.imshow(image)
# plt.show()
#print(predict_image(frame))





images, labels, classes = get_random_images(5)
plt.imshow(np.squeeze(images[0][1,:,:]))
fig=plt.figure(figsize=(10,10))
for ii in range(len(images)):
    #print(images[ii].shape)
    #image = cropped(images[ii])
    #print(image.shape)
    #height, width = images[ii].shape[:2]
    #plt.imshow(image)
    #plt.show()
    image = to_pil(images[ii])

    # left = 0
    # top = height*.8
    # right = width
    # bottom = height
    # image = image.crop((left, top, right, bottom)) 
    #print(image.shape)
    index = predict_image(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()
plt.savefig('result.png')




