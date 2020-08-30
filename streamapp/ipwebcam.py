from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import cv2, os, urllib.request
import numpy as np
from django.conf import settings


# class VideoCamera(object):
# 	def __init__(self):
# 		self.video = cv2.VideoCapture(0)

# 	def __del__(self):
# 		self.video.release()

# 	def get_frame(self):
# 		success, image = self.video.read()
# 		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
# 		# so we must encode it into JPEG in order to correctly display the
# 		# video stream.

# 		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 		faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
# 		for (x, y, w, h) in faces_detected:
# 			cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
# 		frame_flip = cv2.flip(image,1)
# 		ret, jpeg = cv2.imencode('.jpg', frame_flip)
# 		return jpeg.tobytes()


class IPWebCam(object):
    def __init__(self):
        self.url = "http://192.168.0.138:8080/shot.jpg"

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):
        imgResp = urllib.request.urlopen(self.url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

        img = cv2.imdecode(imgNp, -1)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(img, (840, 480), interpolation=cv2.INTER_LINEAR)
        # frame_flip = cv2.flip(resize, 1)

        imgText = cv2.putText(
            img=resize,
            text="Camera 0 (Location)",
            org=(10, 50),  # org Bottom-left corner of the text string in the image
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            thickness=2,  # Thickness of the lines used to draw a text
            color=(209, 80, 0),  # Text color
            fontScale=1
        )

        ret, jpeg = cv2.imencode('.jpg', imgText)

        return jpeg.tobytes()
