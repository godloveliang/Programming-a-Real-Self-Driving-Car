#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

from scipy.spatial import KDTree
import math
import numpy as np

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
	self.base_waypoints = None

	self.waypoints_2d = None
	self.waypoint_tree = None

	#the location of stopline
	self.stopline_2d = [[1148.56, 1184.65],[1559.2, 1158.43],[2122.14, 1526.79],[2175.237, 1795.71],
			    [1493.29, 2947.67],[821.96, 2905.8],[161.76, 2303.82],[351.84, 1574.65]]

	self.stopline_tree = KDTree(self.stopline_2d)

	#set the light's initial state not red
        self.red_light = [-1,-1,-1,-1,-1,-1,-1,-1]

	self.lights = []

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
	rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

	'''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                self.publish_redlight_idx()
            rate.sleep()

    def publish_redlight_idx(self):
	
	#get the closest traffic light's index in stopline_2d
	closest_light_idx =self.get_closest_redlight_idx()

	#get the state of closest light, if it's red light, get it's index in base_waypoint and publish, if not red light publish -1.
	redlight_state = self.red_light[closest_light_idx]

	if redlight_state == 1 and self.waypoint_tree:
	    closest_red_light_idx =self.get_closest_waypoint_idx(closest_light_idx)
	    self.upcoming_red_light_pub.publish(Int32(closest_red_light_idx))
	
	else:
	    self.upcoming_red_light_pub.publish(Int32(-1))

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
	self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
	self.lights = msg.lights

	light_state = [trafficlight.state for trafficlight in msg.lights]

	for i, state in enumerate(light_state):
	    if state == 0:
		self.red_light[i] = 1
	    else:
		self.red_light[i] = -1

    def get_closest_redlight_idx(self):
	#use this function to get the closest traffic light's index in stopline_2d
	x = self.pose.pose.position.x
        y = self.pose.pose.position.y
       
        closest_idx = self.stopline_tree.query([x,y], 1)[1]

        # check if closest is ahead or behind car
        closest_location = self.stopline_2d[closest_idx]
        pre_closest_location = self.stopline_2d[closest_idx-1]

        cl_vect = np.array(closest_location)
        pre_vect = np.array(pre_closest_location)
        pos_vect = np.array([x,y])

        if np.dot(cl_vect - pre_vect, pos_vect - cl_vect) > 0:
            closest_idx = (closest_idx + 1) % len(self.stopline_2d)
        return closest_idx

    def get_closest_waypoint_idx(self, closest_light_idx):
	#use this function to get the redlight's index in waypoints_2d
	x = self.stopline_2d[closest_light_idx][0]
        y = self.stopline_2d[closest_light_idx][1]

        closest_idx = self.waypoint_tree.query([x,y], 1)[1]
        
        # check if closest is ahead or behind car
        closest_waypoint = self.waypoints_2d[closest_idx]
        pre_closest_waypoint = self.waypoints_2d[closest_idx-1]
        
        cl_vect = np.array(closest_waypoint)
        pre_vect = np.array(pre_closest_waypoint)
        pos_vect = np.array([x,y])

        if np.dot(cl_vect - pre_vect, pos_vect - cl_vect) > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
