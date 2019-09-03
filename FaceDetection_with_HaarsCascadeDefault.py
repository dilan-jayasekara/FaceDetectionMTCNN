import cv2
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# load the photograph
pixels = imread('fl.jpg')

# perform face detection using Harrs default face detection model
bboxes = classifier.detectMultiScale(pixels, 1.05, 8)

# print bounding box for each detected face
for box in bboxes:
	# extract
	x, y, width, height = box
	x2, y2 = x + width, y + height
	# draw a rectangle over the pixels
	rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
    
# show the image
imshow('face detection', pixels)

# keep the window open until we press a key
waitKey(0)
# close the window
destroyAllWindows()
