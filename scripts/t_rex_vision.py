#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 15:11:15 2021

@author: iad
"""
import rospy
from multiprocessing import Lock
import cv2, cv_bridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import numpy as np
import imutils
import homography

def somefunct(arg):
    out = arg
    return out

class MovingObjectDetector:
    def __init__(self):

        rospy.init_node("moving_object_detector", anonymous=False)
        # parameters from the ros parameter server
        self.read_parameters()
        # subscriber
        self.image_sub = rospy.Subscriber('image_raw', Image, self.image_callback)
        # publisher
        self.moving_pub = rospy.Publisher('moving_object', Image, queue_size=1)
        # timer
        self.processing_timer = rospy.Timer(rospy.Duration(0.05), processing)
        # cv_bridge
        self.bridge = cv_bridge.CvBridge()
        # class member
        self.isNewImage_ = False
        self.imageNow_ = np.array([])
        self.imagePrev_ = np.array([])
        self.imageNowCopy_ = imageNow_.copy()
        self.imagePrevCopy_ = imagePrev_.copy()
        self.mutex = Lock()
        

    def read_parameters(self):
        self.param = {}
        self.param['ransac'] = {}
        # ransac
        self.param['ransac']['threshold'] = rospy.get_param('~ransac/threshold', 1.0)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~ransac/threshold'), self.param['ransac']['threshold'])
        self.param['ransac']['sample_size'] = rospy.get_param('~ransac/sample_size', 5)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~ransac/sample_size'), self.param['ransac']['sample_size'])
        self.param['ransac']['goal_inliers'] = rospy.get_param('~ransac/goal_inliers', 0.6)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~ransac/goal_inliers'), self.param['ransac']['goal_inliers'])
        self.param['ransac']['max_iterations'] = rospy.get_param('~ransac/max_iterations', 100)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~ransac/max_iterations'), self.param['ransac']['max_iterations'])
        self.param['ransac']['stop_at_goal'] = rospy.get_param('~ransac/stop_at_goal', True)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~ransac/stop_at_goal'), self.param['ransac']['stop_at_goal'])
        self.param['ransac']['random_seed'] = rospy.get_param('~ransac/random_seed', None)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~ransac/random_seed'), self.param['ransac']['random_seed'])
        # other params
        self.param['downsamplingStep'] = rospy.get_param('~downsamplingStep', 20)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~downsamplingStep'), self.param['downsamplingStep'])

    def image_callback(self,msg):
        # conversion ros image vers opencv
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')

        cv2.imshow("Lecture image du drone", image)
        cv2.waitKey(1)
        # conversion et publication image opencv vers message ros
        self.moving_pub.publish(self.bridge.cv2_to_imgmsg(image, encoding="passthrough"))
        
    def processing(self):
        # nothing
        return 0
        

if __name__ == '__main__':
    try:
        movingObjectDetector = MovingObjectDetector()
    except rospy.ROSInterruptException: pass
    rospy.spin()