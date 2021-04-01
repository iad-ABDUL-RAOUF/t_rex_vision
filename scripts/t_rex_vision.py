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
import homography as hm


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
        self.processing_timer = rospy.Timer(rospy.Duration(0.05), self.processing)
        # cv_bridge
        self.bridge = cv_bridge.CvBridge()
        # class member
        self.count = self.param['stepImage']
        self.isImage_ = False
        self.isNewImage_ = False
        self.isProcessInit_ = False
        self.imageNow_ = None
        self.imagePrev_ = None
        self.imageNowProcess_ = None
        self.imagePrevProcess_ = None
        self.mutex_ = Lock()
        

    def read_parameters(self):
        self.param = {}
        self.param['ransac'] = {}
        # ransac TODO virer param inutile
        self.param['ransac']['threshold'] = rospy.get_param('~ransac/threshold', 15.0)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~ransac/threshold'), self.param['ransac']['threshold'])
        self.param['ransac']['sample_size'] = rospy.get_param('~ransac/sample_size', 5)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~ransac/sample_size'), self.param['ransac']['sample_size'])
        self.param['ransac']['goal_inliers'] = rospy.get_param('~ransac/goal_inliers', 0.6)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~ransac/goal_inliers'), self.param['ransac']['goal_inliers'])
        self.param['ransac']['max_iterations'] = rospy.get_param('~ransac/max_iterations', 100)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~ransac/max_iterations'), self.param['ransac']['max_iterations'])
        self.param['ransac']['stop_at_goal'] = rospy.get_param('~ransac/stop_at_goal', True)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~ransac/stop_at_goal'), self.param['ransac']['stop_at_goal'])
        # other params
        self.param['step'] = rospy.get_param('~downsamplingStep', 20)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~downsamplingStep'), self.param['step'])
        self.param['stepImage'] = rospy.get_param('~consecutiveImageStep', 8)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~consecutiveImageStep'), self.param['stepImage'])
        # crop
        self.param['crophmin'] = rospy.get_param('~crophmin', 200)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~crophmin'), self.param['crophmin'])
        self.param['crophmax'] = rospy.get_param('~crophmax', 0)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~crophmax'), self.param['crophmax'])
        self.param['cropwmin'] = rospy.get_param('~cropwmin', 0)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~cropwmin'), self.param['cropwmin'])
        self.param['cropwmax'] = rospy.get_param('~cropwmax', 0)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~cropwmax'), self.param['cropwmax'])


    def image_callback(self,msg):
        self.count = self.count +1
        if self.param['stepImage'] <= self.count:
            with self.mutex_:
                try:
                    self.imagePrev_ = self.imageNow_.copy()
                    self.isImage_ = True
                except AttributeError:
                    rospy.logwarn("imageNow_ has no attribute 'copy'. Probably still at init value 'None'")
                # conversion ros image vers opencv
                self.imageRead_ = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
                shape = self.imageRead_.shape
                self.imageNow_ = self.imageRead_[self.param['crophmin']:shape[0]-self.param['crophmax'],self.param['cropwmin']:shape[1]-self.param['cropwmax']]
                self.isNewImage_ = True
            self.count = 0

        # cv2.imshow("debug self.imageNow_", self.imageNow_)
        # cv2.imshow("debug self.imagePrev_", self.imagePrev_)
        # cv2.waitKey(1)
        
    def processing(self, event):
        if (self.isNewImage_ and self.isImage_):
            # load images
            with self.mutex_:
                self.imageNowProcess_ = cv2.cvtColor(self.imageNow_,cv2.COLOR_BGR2GRAY)
                self.imagePrevProcess_ = cv2.cvtColor(self.imagePrev_,cv2.COLOR_BGR2GRAY)
                self.isNewImage_ = False
            
            # init some processing variable
            if not(self.isProcessInit_):
                self.initBeforeFirstProcess()
                self.isProcessInit_ = True
            
            # do processing
            # compute optical flow. TODO put optical flow parameters in the parameter server
            self.flow = cv2.calcOpticalFlowFarneback(   self.imagePrevProcess_,self.imageNowProcess_,None, 
                                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                                        levels = 3, # Nombre de niveaux de la pyramide
                                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                                        iterations = 3, # Nb d'itérations par niveau
                                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                                        flags = 0)
            # compute down sampled pixel coordinates after they are moved by the optical flow
            self.ds_h_movedPixels = self.ds_h_pixels + np.concatenate(( self.flow[self.step//2::self.step,self.step//2::self.step,0].reshape((-1,1)),
                                                                        self.flow[self.step//2::self.step,self.step//2::self.step,1].reshape((-1,1)),
                                                                        np.zeros((self.ds_h_pixels.shape[0],1))),
                                                                        axis = 1)
            # compute homography linking pixel before and after they moved. Correspond to global picture mouvement i.e. ego-motion 
            # homography, ds_nIinliers = hm.run_ransac(  self.ds_h_pixels,
            #                                         self.ds_h_movedPixels,
            #                                         self.param['ransac']['threshold'],
            #                                         self.param['ransac']['sample_size'],
            #                                         self.goal_inliers,
            #                                         self.param['ransac']['max_iterations'],
            #                                         self.param['ransac']['stop_at_goal'],
            #                                         None)[0:2]
            homography = cv2.findHomography(self.ds_h_pixels, self.ds_h_movedPixels, cv2.RANSAC, self.param['ransac']['threshold'])[0]
            if homography.size == 0:
                rospy.logwarn('ego motion model estimation failed')
            else:
                # compute outliers in the full image
                self.h_movedPixels = self.h_pixels + np.concatenate((   self.flow[:,:,0].reshape((-1,1)),
                                                                        self.flow[:,:,1].reshape((-1,1)),
                                                                        np.zeros((self.h*self.w,1))),
                                                                    axis = 1)
                self.moving_ = np.logical_not(hm.findHomographyInlier(  homography,
                                                                        self.h_pixels,
                                                                        self.h_movedPixels,
                                                                        self.param['ransac']['threshold']).reshape((self.h, self.w)))
                rospy.loginfo('%s inliers, %s outliers (moving pixel)', self.moving_.sum(), self.h*self.w - self.moving_.sum())

                # conversion and publication of the moving pixels
                self.moving_pub.publish(self.bridge.cv2_to_imgmsg(self.moving_.astype(np.uint8)*255, encoding="mono8"))  

                # graphical display
                self.mag, self.ang = cv2.cartToPolar(self.flow[:,:,0], self.flow[:,:,1])
                self.hsv[:,:,0] = (self.ang*180)/(2*np.pi)
                self.hsv[:,:,2] = (self.mag*255)/np.amax(self.mag)
                self.bgr = cv2.cvtColor(self.hsv,cv2.COLOR_HSV2BGR)
                # cv2.imshow("debug self.imageNowProcess_", self.imageNowProcess_)
                # cv2.imshow("debug self.imagePrevProcess_", self.imagePrevProcess_)
                # cv2.imshow('mooving object',self.moving_.astype(np.uint8)*255)
                cv2.imshow('self.imagePrevProcess_',self.imagePrevProcess_)
                cv2.imshow('self.imageNowProcess_',self.imageNowProcess_)
                cv2.imshow('optical flow',self.bgr)
                cv2.waitKey(1)
                
                
    
    def initBeforeFirstProcess(self):
        (self.h,self.w) = self.imageNowProcess_.shape
        self.step = self.param['step']
        # homogeneous pixel list 
        hMat, wMat = np.meshgrid(np.arange(self.h), np.arange(self.w), indexing = 'ij')
        self.h_pixels = np.concatenate((hMat.reshape((-1,1)),
                                        wMat.reshape((-1,1)),
                                        np.ones((self.h*self.w,1))),
                                        axis = 1)
        # down sampled homogeneous pixel list 
        ds_hRange = np.arange(self.step//2,self.h,self.step)
        ds_wRange = np.arange(self.step//2,self.w,self.step)
        hMat, wMat = np.meshgrid(ds_hRange, ds_wRange, indexing = 'ij')
        self.ds_h_pixels = np.concatenate(( hMat.reshape((-1,1)),
                                            wMat.reshape((-1,1)),
                                            np.ones((ds_hRange.size*ds_wRange.size,1))),
                                        axis = 1)
        # ransac param
        self.goal_inliers = int(ds_hRange.size*ds_wRange.size*self.param['ransac']['goal_inliers'])
        # graphical display
        self.hsv = np.zeros([self.h,self.w,3],dtype=np.uint8)
        self.hsv[:,:,1] = 255 # color saturation

        

if __name__ == '__main__':
    try:
        movingObjectDetector = MovingObjectDetector()
    except rospy.ROSInterruptException: pass
    rospy.spin()