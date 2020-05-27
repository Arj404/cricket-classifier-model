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


def ocr(image):
	image = cv2.imread(image)
	image = ResizeWithAspectRatio(image, width=960)
	print(image.shape[:2])
	#image = masking(image,150,0,530,960)
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
			# if '/' in text:
			# 	s = ''
			# 	for i in text:
			# 		l=['0','1','2','3','4','5','6','7','8','9','/']
			# 		if i in l:
			# 			s = s+i
			# 	s = text.split('/')
			# 	F_text.append(s)
			# 	#print(s)
			# if '-' in text:
			# 	s = ''
			# 	for i in text:
			# 		l=['0','1','2','3','4','5','6','7','8','9','-']
			# 		if i in l:
			# 			s = s+i
			# 	s = text.split('-')
			# 	F_text.append(s)
			# 	#print(s)
			# if '.' in text:
			# 	s = ''
			# 	for i in text:
			# 		l=['0','1','2','3','4','5','6','7','8','9','.']
			# 		if i in l:
			# 			s = s+i
			F_text.append(text)
			#print("Confidence: {}".format(conf))
			#print("Text: {}".format(text))
			#print("")
			text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
			cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
				.7, (0, 0, 255), 1)

	cv2.imshow("Image", image)
	cv2.waitKey(0)
	return F_text


text = ocr('./Screen.png')
print(text)