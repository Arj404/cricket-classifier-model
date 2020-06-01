from pytesseract import Output
import pytesseract
import argparse
import cv2
import numpy as np


def masking(image,xl,yl,xr,yr):
	mask = np.zeros(image.shape[:2], np.uint8)
	mask[xl:xr,yl:yr] = 1
	image = image*mask[:,:,np.newaxis]
	return image


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


def ocr(image,xl,yl,xr,yr):
	image = cv2.imread(image)
	image = ResizeWithAspectRatio(image, width=960)
	#image = masking(image,470,190,500,260)
	image = image[xl:xr,yl:yr]
	image = cv2.resize(image, None, fx=2, fy=2)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	results = pytesseract.image_to_data(rgb, output_type=Output.DICT)
	F_text = []

	for i in range(0, len(results["text"])):
		x = results["left"][i]
		y = results["top"][i]
		w = results["width"][i]
		h = results["height"][i]
		text = results["text"][i]
		conf = int(results["conf"][i])
		if conf > 0:
			F_text.append(text)
	# 		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	# 		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
	# 		cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
	# 			1, (0, 0, 255), 1)

	# cv2.imshow("Image", image)
	# cv2.waitKey(0)
	return F_text


def RunOcr(image_path):
	Score_Wicket = ocr(image_path,470,195,500,265)
	s = ''
	Wicket = ''
	Score = ''
	if Score_Wicket:
		if '/' in Score_Wicket[0]:
			for i in Score_Wicket[0]:
				l=['0','1','2','3','4','5','6','7','8','9','/']
				if i in l:
					s = s+i
	
	
	if s:
		Wicket,Score = s.split('/')
	
	
	overs = ocr(image_path,475,285,495,330)
	Overs = ''
	if overs:
		if overs[0].isdecimal():
			Overs = overs[0]
	
	
	Batsman1 = ocr(image_path,496,332,520,472)
	Bat1 = ''
	if Batsman1:
		if len(Batsman1) > 1:
			if Batsman1[1]:
				if Batsman1[1].isalpha():
					Bat1 = Batsman1[1]
	
	
	Batsman2 = ocr(image_path,496,500,520,650)
	Bat2 = ''
	if Batsman2:
		if len(Batsman2) > 1:
			if Batsman2[1]:
				if Batsman2[1].isalpha():
					Bat2 = Batsman2[1]
	
	
	current=0
	if Batsman2[0] == '»':
		current=2
	elif Batsman1[0] == '»':
		current=1
	return Score,Wicket,Overs,Bat1,Bat2,current


image_path = './Screen1.png'
Score,Wicket,Overs,Bat1,Bat2,current = RunOcr(image_path)

print(Score)
print(Wicket)
print(Overs)
print(Bat1)
print(Bat2)
print(current)

