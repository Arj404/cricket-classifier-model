import cv2
import time
import copy
import ocr
import pandas as pd
start_time = time.time()

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]

	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))

	return cv2.resize(image, dim, interpolation=inter)



def video_capture(url,f):
	cap = cv2.VideoCapture(url)
	count = 0
	f_list = []
	height = 0
	width = 0
	while cap.isOpened():
		ret,frame = cap.read()
		if ret == True:

			if(count%f==0):
				r_frame = ResizeWithAspectRatio(frame, height=1000)
				f_list.append(r_frame)
			count = count + 1
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
		else:
			break
	cap.release()
	return f_list


def video_generate(f_list,name,fps):
	(he, wi) = f_list[0].shape[:2]
	out = cv2.VideoWriter(name,0,fps, (wi,he))
	for i in range(len(f_list)):
		out.write(f_list[i])
	out.release()


def live_detection(f_list):
	image_template1 = cv2.imread('image.png')
	image_template1 = cv2.cvtColor(image_template1, cv2.COLOR_BGR2GRAY)
	x_list = []
	valuesl = []
	values = []
	c=0
	for frame in f_list:
		height, width = frame.shape[:2]
		im = copy.copy(frame)
		top_left_x,top_left_y = int(height * .8), int(width * 0)
		bottom_right_x,bottom_right_y = int(height*1), int(width*1)

		cv2.rectangle(frame, (top_left_y,top_left_x), (bottom_right_y,bottom_right_x), (127,50,127), 3)
		cropped = frame[top_left_x:bottom_right_x , top_left_y:bottom_right_y]

		image1 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
		orb1 = cv2.ORB_create(1000, 1.2, 8, 15, 0,2)
		kp1, des1 = orb1.detectAndCompute(image1, None)
		kp2, des2 = orb1.detectAndCompute(image_template1, None)
		bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches1 = bf1.match(des1,des2)
		matches1 = sorted(matches1, key=lambda val: val.distance)
		matches1 = len(matches1)
		x = ''
		output_string = str(matches1)
		cv2.putText(frame, output_string, (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
		threshold = 290
		if matches1 > threshold:
			x = 'L'
			cv2.putText(frame,'L',(140,20), cv2.FONT_HERSHEY_COMPLEX, 2 ,(255,0,0), 2)
			Score,Wicket,Overs,Bat1,Bat2,current = ocr.RunOcr(frame)
			valuesl.append([Score,Wicket,Overs,Bat1,Bat2,current])
		else:
			x = 'R'
			cv2.putText(frame,'R',(140,20), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,0,255), 2)

		Score,Wicket,Overs,Bat1,Bat2,current = ocr.RunOcr(frame)
		values.append([x,Score,Wicket,Overs,Bat1,Bat2,current])
		c=c+1
		if(c%70==0):
			print(c)
		x_list.append(frame)
	return x_list, valuesl, values


v_url = './video.avi'
f_list = []
p_list = []
valuesl = []
values = []
f_list= video_capture(v_url,1)
print(len(f_list))


p_list, valuesl, values = live_detection(f_list)

df = pd.DataFrame(data=valuesl)
df.to_csv("./filel.csv", sep=',',index=False)

df2 = pd.DataFrame(data=values)
df2.to_csv("./file.csv", sep=',',index=False)

#video_generate(f_list,'video_processed.avi',30)

cv2.destroyAllWindows()
print("--- %s seconds ---" % (time.time() - start_time))