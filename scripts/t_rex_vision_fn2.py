#!/usr/bin/env python3
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

# for flownet2 with tensorpack
import argparse
import cv2
from tensorpack import *
from tensorpack.utils import viz
import flownet_models as models
from helper import Flow




class Flownet2Algorithm:
    def __init__(self, model, model_path, imgH, imgW):
        newh = (imgH // 64) * 64
        neww = (imgW // 64) * 64
        self.aug = imgaug.CenterCrop((newh, neww))
        self.predict_func = OfflinePredictor(PredictConfig(
            model=model(height=newh, width=neww),
            session_init=SmartInit(model_path),
            input_names=['left', 'right'],
            output_names=['prediction']))
        self.flow = Flow()

    def apply(self, imgPrev, imgNow):
        imgPrev = self.aug.augment(imgPrev)
        imgNow = self.aug.augment(imgNow)
        inputPrev, inputNow = [x.astype('float32').transpose(2, 0, 1)[None, ...]
                                for x in [imgPrev, imgNow]]
        output = self.predict_func(inputPrev, inputNow)[0].transpose(0, 2, 3, 1)
        imgOut = self.flow.visualize(output[0])
        patches = [imgPrev, imgNow, imgOut * 255.]
        imgOut = viz.stack_patches(patches, 2, 2)
        cv2.imshow('flow output', imgOut)
        cv2.imwrite('flow_output.png', imgOut)
        cv2.waitKey(0)
        return imgOut



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
        # ransac
        self.param['ransac'] = {}
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
        # flownet2
        self.param['flownet2'] = {}
        self.param['flownet2']['load'] = rospy.get_param('~load')
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~load'), self.param['flownet2']['load'])
        self.param['flownet2']['model'] = rospy.get_param('~model')
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~model'), self.param['flownet2']['model'])
        # other params
        self.param['step'] = rospy.get_param('~downsamplingStep', 20)
        rospy.loginfo('Parameter %s has value %s', rospy.resolve_name('~downsamplingStep'), self.param['step'])

    def image_callback(self,msg):
        with self.mutex_:
            try:
                self.imagePrev_ = self.imageNow_.copy()
                self.isImage_ = True
            except AttributeError:
                rospy.logwarn("imageNow_ has no attribute 'copy'. Probably still at init value 'None'")
            # conversion ros image vers opencv
            self.imageNow_ = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            self.isNewImage_ = True

        # cv2.imshow("debug self.imageNow_", self.imageNow_)
        # cv2.imshow("debug self.imagePrev_", self.imagePrev_)
        cv2.waitKey(1)
        
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
            
            # do processing
            # compute optical flow. 
            self.flow = self.fn2Algorithm.apply(self.imagePrevProcess_, self.imageNowProcess_)
            # TODO put farneback parameters in the parameter server
            # self.flow = cv2.calcOpticalFlowFarneback(   self.imagePrevProcess_,self.imageNowProcess_,None, 
            #                                             pyr_scale = 0.5,# Taux de réduction pyramidal
            #                                             levels = 3, # Nombre de niveaux de la pyramide
            #                                             winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
            #                                             iterations = 3, # Nb d'itérations par niveau
            #                                             poly_n = 7, # Taille voisinage pour approximation polynomiale
            #                                             poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
            #                                             flags = 0)
            # compute down sampled pixel coordinates after they are moved by the optical flow
            self.ds_h_movedPixels = self.ds_h_pixels + np.concatenate(( self.flow[self.step//2::self.step,self.step//2::self.step,0].reshape((-1,1)),
                                                                        self.flow[self.step//2::self.step,self.step//2::self.step,1].reshape((-1,1)),
                                                                        np.zeros((self.ds_h_pixels.shape[0],1))),
                                                                        axis = 1)
            # compute homography linking pixel before and after they moved. Correspond to global picture mouvement i.e. ego-motion 
            homography, ds_nIinliers = hm.run_ransac(  self.ds_h_pixels,
                                                    self.ds_h_movedPixels,
                                                    self.param['ransac']['threshold'],
                                                    self.param['ransac']['sample_size'],
                                                    self.goal_inliers,
                                                    self.param['ransac']['max_iterations'],
                                                    self.param['ransac']['stop_at_goal'],
                                                    None)[0:2]
            if ds_nIinliers == None:
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
                # cv2.imshow('optical flow',self.bgr)
                # cv2.waitKey(1)
                
                
    
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

        # optical flow algorithm
        model = {'flownet2-s': models.FlowNet2S,
             'flownet2-c': models.FlowNet2C,
             'flownet2': models.FlowNet2}[self.param['flownet2']['model']]
        self.fn2Algorithm = Flownet2Algorithm(model, self.param['flownet2']['load'], self.h, self.w)
        

if __name__ == '__main__':
    try:
        movingObjectDetector = MovingObjectDetector()
    except rospy.ROSInterruptException: pass
    rospy.spin()






# debut d'implentation d'un tuto (A supprimer)

# from ros_tensorflow.model import ModelWrapper, StopTrainOnCancel, EpochCallback

# class RosInterface():
#     def __init__(self):
#         # TODO model
#         self.input_dim = 10
#         self.output_dim = 2
#         self.wrapped_model = ModelWrapper(input_dim=self.input_dim, output_dim=self.output_dim)

#         # self.predict_srv = rospy.Service('predict', Predict, self.predict_cb)
#         # self.predict_pub ... TODO

#     def predict_cb(self, req):
#         # TODO transformer en fonction normale prenant sans arguments (allant directment chercher les images membre de la classe)
#         rospy.loginfo("Prediction from service")
#         x = np.array(req.data).reshape(-1, self.input_dim)
#         i_class, confidence = self.wrapped_model.predict(x)
#         return PredictResponse(i_class=i_class, confidence=confidence)