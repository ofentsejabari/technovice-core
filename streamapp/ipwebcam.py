from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import cv2, os, urllib.request
import numpy as np
from django.conf import settings

# from mrcnn.config import Config
# from mrcnn.model import MaskRCNN

# import streamapp.visualize_cv2 as vlc


# class TestConfig(Config):
#     NAME = "test"
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     NUM_CLASSES = 1 + 80


# rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())

# rcnn.load_weights('/home/ojabari/.keras/models/mask_rcnn_coco.h5', by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


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
        self.url = "http://192.168.81.85:8080/shot.jpg"

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

        # -- Run mask RCNN object detection on the acquired frame --
        # results = rcnn.detect([resize], verbose=0)
        # r = results[0]

        # frame = vlc.display_instances(
        #     frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        # )
        
        # -- Labeled frame --
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
