from styx_msgs.msg import TrafficLight
from keras.models import load_model
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
	   # Assumes that the keras file is in the same folder
        self.model = load_model('tl_keras_carBv03.h5')
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        imgdisp = image[:,:,::-1] # Convert to RGB
        imgpred = imgdisp.reshape((1,imgdisp.shape[0], imgdisp.shape[1], imgdisp.shape[2]))
        classpred = model.predict_classes(imgpred)
        output = TrafficLight.Red if classpred[0]==0 else TrafficLight.UNKNOWN
        return output