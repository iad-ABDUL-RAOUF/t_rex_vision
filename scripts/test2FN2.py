#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# for flownet2 with tensorpack
import argparse
import cv2
from tensorpack import *
from tensorpack.utils import viz
import flownet_models as models
from helper import Flow

import rospy


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

if __name__ == '__main__':
    rospy.init_node("testFlownet2", anonymous=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='path to the model', required=True)
    parser.add_argument('--model', help='model',
                        choices=['flownet2', 'flownet2-s', 'flownet2-c'], required=True)
    parser.add_argument('--images', nargs="+",
                        help='a list of equally-sized images. FlowNet will be applied to all consecutive pairs')
    args = parser.parse_args()

    model = {'flownet2-s': models.FlowNet2S,
             'flownet2-c': models.FlowNet2C,
             'flownet2': models.FlowNet2}[args.model]
    assert len(args.images) >= 2
    
    imgPrev = cv2.imread(args.images[0])
    imgNow = cv2.imread(args.images[1])
    h, w = imgPrev.shape[:2]
    fn2Algorithm = Flownet2Algorithm(model, args.load, h, w)
    fn2Algorithm.apply(imgPrev, imgNow)