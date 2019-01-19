
import tensorflow as tf
import numpy as np
import time

if __name__ != '__main__':
    from styx_msgs.msg import TrafficLight

#===================================================================================
# Utility Functions

#-------------------------------------------------------------------------
def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

#-------------------------------------------------------------------------
def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

#-------------------------------------------------------------------------
def convert_to_image_coordinates(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords

#-------------------------------------------------------------------------
from PIL import ImageDraw
from PIL import ImageColor
# Colors (one for each class)
cmap = ImageColor.colormap
#print("Number of colors =", len(cmap))
COLOR_LIST = sorted([c for c in cmap.keys()])

def draw_boxes(image, boxes, classes, thickness=4):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

#-------------------------------------------------------------------------
def convert_enum_traffic_light_color(classes, detected_colour):
    pass
    

#===================================================================================
class TLClassifier(object):
    def __init__(self, boIsContextRealCar, boDebugMode, ConfidenceThreshold):
    
        self.graph = tf.Graph()
        self.detection_confidence = ConfidenceThreshold
        self.is_context_real_car = boIsContextRealCar
        self.is_debug_mode = boDebugMode
        
        #TODO load classifier
        if (self.is_context_real_car==False):
            GraphFilePath_SSD = 'light_classification/model_udsim/ssd_inception_v2_inference_graph.pb'
            ##GraphFilePath_SSD = "models/frozen_ssd_sim_20190114_10K_Steps/frozen_inference_graph.pb"'
            ##GraphFilePath_SSD = 'light_classification/model_udsim/frozen_inference_graph.pb'
            self.detection_graph = load_graph(GraphFilePath_SSD)

            # The input placeholder for the image.
            # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

            # The classification of the object (integer id).
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

            self.sess = tf.Session(graph=self.detection_graph)
        else:
            pass
            
#-----------------------------------------------------------------------------------
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        with self.detection_graph.as_default():
            if(self.is_debug_mode == True):
                ### For Test Only
                from PIL import Image
                image_sample = Image.open(image)
                image_np = np.expand_dims(np.asarray(image_sample, dtype=np.uint8), 0)
            else:
                image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

            # Actual detection.
            time_start = time.time()
            (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], \
                                                        feed_dict={self.image_tensor: image_np})
            time_end = time.time()
            detection_time = (time_end - time_start)##*1000
            
        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)##.astype(np.int32)

        print()
        print("Detection Time(sec):{0:f} | Score:{1:f} | Class:{2:f}".format(detection_time,  scores[0],  classes[0]) )
        ##print("Detection Time(sec):", detection_time)
        ##print("Score:", scores)
        ##print("Class", classes)
    
        if(self.is_debug_mode == True):

            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = filter_boxes(self.detection_confidence, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            width, height = image_sample.size
            box_coords = convert_to_image_coordinates(boxes, height, width)

            # Each class with be represented by a differently colored box
            draw_boxes(image_sample, box_coords, classes)
            mpimg.imsave("./processed_image/processed_"+image+".png", image_sample, format='png')
            ##plt.style.use('ggplot')
            ##plt.figure(figsize=(12, 8))
            ##plt.imshow(image_sample)

        classes = classes.astype(np.int32)
        
        detected_colour = 999
        
        if (scores[0] > self.detection_confidence):
            if (classes[0] == 2):
                print('Red')
                if __name__ != '__main__': detected_colour = TrafficLight.RED
            elif (classes[0] == 3):
                print('Yellow')
                if __name__ != '__main__': detected_colour = TrafficLight.YELLOW
            elif (classes[0] == 1):
                print('Green')
                if __name__ != '__main__': detected_colour = TrafficLight.GREEN
            else:
                print('No Traffic Light')
                if __name__ != '__main__': detected_colour = TrafficLight.UNKNOWN
        else:
            print('No Traffic Light')
            if __name__ != '__main__': detected_colour = TrafficLight.UNKNOWN
        
        return detected_colour
        
#===================================================================================
if __name__ == '__main__':

    light_classifier = TLClassifier(boIsContextRealCar  = False, 
                                    boDebugMode         = True, 
                                    ConfidenceThreshold = 0.5)
                                    
    light_classifier.get_classification("green1.jpg")
    light_classifier.get_classification("green2.jpg")
    light_classifier.get_classification("red1.jpg")
    light_classifier.get_classification("red2.jpg")
    light_classifier.get_classification("yellow1.jpg")
    light_classifier.get_classification("yellow2.jpg")