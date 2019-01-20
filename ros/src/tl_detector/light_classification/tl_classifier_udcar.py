from styx_msgs.msg import TrafficLight
from keras.models import load_model
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
	   # Assumes that the keras file is in the same folder
        model_path = "light_classification/model_udcar/tl_keras_carbv03.h5"
        self.model = load_model(model_path)
        
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
        classpred = self.model.predict_classes(imgpred)
        output = TrafficLight.RED if classpred[0]==0 else TrafficLight.UNKNOWN
        #print("Predicted class", classpred[0])
        #print("Output", output)
        return output