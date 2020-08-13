from PIL import Image

import numpy as np

from rest_framework.views import APIView
from rest_framework.exceptions import ParseError
from rest_framework.response import Response

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16

import pickle


class Animal10Predict(APIView):
    """
    A convolution Neural network multiclass classifier trained on animal10 dataset. The model
    given an image of either [cat, dog, ] will return the label or identity of that animal.
    """

    def post(self, request, format=None):
        if 'file' not in request.data:
            raise ParseError("Empty content")

        print(request)

        f = request.data['file']
        meta = request.data['meta']

        try:
            img = Image.open(f)
            # img.save(f'{f}')
            # print(img)
            img.verify()
        except:
            raise ParseError("Unsupported image type")

        # Instantiating the VGG16 convolutional base
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
        # -- Load our custom classifier
        trained_model = load_model('/technovice/animals10-feature-extraction_vgg16.h5')

        with open('/technovice/classes-animals10.pickle', 'rb') as handler:
            animal10_class_index = pickle.load(handler)

        # -- Load the image --
        img = load_img(f, target_size=(150, 150, 3))
        # -- Convert image pixels to a numpy array --
        img = img_to_array(img)
        # -- Expected model input: (n_samples, 150, 150, 3) --
        img = np.expand_dims(img, axis=0)
        img /= 255.

        # -- Extract features --
        conv_base_output = conv_base.predict(img)

        '''
            Run the extracted features on our trained classifier .
            Predict probabilities across all 10 output classes.
            classifier_output: [
                [4.76799974e-12 9.99999999e-11 ...]
            ]
        '''
        classifier_output = trained_model.predict(conv_base_output)[0]

        # -- Convert the probabilities to class labels
        class_prop = {}
        for i in range(len(classifier_output)):
            if np.round(classifier_output[i], 3) >= .50:
                class_prop[list(animal10_class_index)[i]] = classifier_output[i]

        if len(class_prop) > 0:
            # -- Sort the dictionary in descending order of probabilities --
            sorted_class_prob = [{'class': w, 'value': class_prop[w]} for w in
                                 sorted(class_prop, key=class_prop.get, reverse=True)]
            response_status = 'ACCEPTED'

        else:
            sorted_class_prob = 'Sorry! The model confidence on this image is very low.'
            response_status = 'NOT_ACCEPTED'

        print({'response': sorted_class_prob, 'status': response_status})
        # -- Retrieve the most likely n results i.e. Highest probability --
        # -- sorted_class_prob = {k: sorted_class_prob[k] for k in list(sorted_class-prob)[:n]}
        return Response(data={'response': sorted_class_prob, 'status': response_status})


class VGG16APIFX(APIView):
    """ """

    def post(self, request, format=None):

        if 'file' not in request.data:
            raise ParseError("Empty content")

        f = request.data['file']
        meta = request.data['meta']

        try:
            img = Image.open(f)
            img.verify()
        except:
            raise ParseError("Unsupported image type")

        # load the model
        model = VGG16()
        # load an image from file
        image = load_img(f, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # predict the probability across all output classes
        yhat = model.predict(image)
        # convert the probabilities to class labels
        label = decode_predictions(yhat)
        print(label)
        # retrieve the most likely result, e.g. highest probability
        label = label[:2]
        # print the classification
        # print('%s (%.2f%%)' % (label[1], label[2]*100))
        return Response(label)
