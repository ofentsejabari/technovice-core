from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from streamapp.ipwebcam import IPWebCam


def index(request):
    return render(request, 'streamapp/index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# def video_feed(request):
# 	return StreamingHttpResponse(gen(VideoCamera()),
# 					content_type='multipart/x-mixed-replace; boundary=frame')


# def webcam_feed(request):
# 	return StreamingHttpResponse(gen(IPWebCam()),
# 					content_type='multipart/x-mixed-replace; boundary=frame')


# def mask_feed(request):
# 	return StreamingHttpResponse(gen(MaskDetect()),
# 					content_type='multipart/x-mixed-replace; boundary=frame')

def livecam_feed(request):
    return StreamingHttpResponse(gen(IPWebCam()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


from PIL import Image

import numpy as np

from rest_framework.views import APIView
from rest_framework.exceptions import ParseError
from rest_framework.response import Response


class LiveFeed(APIView):

    def get(self, request, format=None):
        return StreamingHttpResponse(gen(IPWebCam()),
                                     content_type='multipart/x-mixed-replace; boundary=frame')
