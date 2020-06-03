import cv2


def video_generate(f_list):
	(he, wi) = f_list[0].shape[:2]
	out = cv2.VideoWriter('video.avi',0,30, (wi,he))
	for i in range(len(f_list)):
		out.write(f_list[i])
	out.release()


def video_capture(url):
	cap = cv2.VideoCapture(url)
	count = 0
	f_list = []
	while cap.isOpened():
		ret,frame = cap.read()
		if ret == True:
			if count%500==0:
				f_list.append(frame)
			count = count + 1
			if (cv2.waitKey(10) & 0xFF == ord('q')):
				break
		else:
			break
	cap.release()
	return f_list

v_url = './Video.mp4'
frame_list = []
frame_list = video_capture(v_url)
print(len(frame_list))
video_generate(frame_list)