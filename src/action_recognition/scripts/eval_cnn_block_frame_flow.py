#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy
import torch
from models.cnn_block_frame_flow import CNNBlockFrameFlow
from torch.autograd import Variable
from std_msgs.msg import String
from sensor_msgs.msg import Image 
from sensor_msgs.msg import CompressedImage
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import sys, os
import time

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
cv_bridge_path = '/home/nvidia/Desktop/ti_ros/devel/lib/python3/dist-packages'
if cv_bridge_path not in sys.path:
    sys.path.append(cv_bridge_path)
# print(sys.path)
from cv_bridge import CvBridge, CvBridgeError
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


CATEGORIES = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running", 
    "walking"
]


def getscore(model, current_block_frame, current_block_flow_x, current_block_flow_y, mean, use_cuda):

    block_frame = np.array(
        current_block_frame, dtype=np.float32).reshape((1, 15, 60, 80))

    block_flow_x = np.array(
        current_block_flow_x, dtype=np.float32).reshape((1, 14, 30, 40))

    block_flow_y = np.array(
        current_block_flow_y, dtype=np.float32).reshape((1, 14, 30, 40))

    block_frame -= mean["frames"]
    block_flow_x -= mean["flow_x"]
    block_flow_y -= mean["flow_y"]

    tensor_frames = torch.from_numpy(block_frame)
    tensor_flow_x = torch.from_numpy(block_flow_x)
    tensor_flow_y = torch.from_numpy(block_flow_y)

    instance_frames = Variable(tensor_frames.unsqueeze(0))
    instance_flow_x = Variable(tensor_flow_x.unsqueeze(0))
    instance_flow_y = Variable(tensor_flow_y.unsqueeze(0))

    if use_cuda == True:
        instance_frames = instance_frames.cuda()
        instance_flow_x = instance_flow_x.cuda()
        instance_flow_y = instance_flow_y.cuda()                  

    score = model(instance_frames, instance_flow_x, instance_flow_y).data[0].cpu().numpy()
    return score



class action_recogintion:
    def __init__(self, model_dir, use_cuda, block_num):
        self.use_cuda = use_cuda
        self.block_num = block_num
        self.data = []      # The images and optical flows data
        self.current_block_image = []
        self.current_block_flow_x = []
        self.current_block_flow_y = []    
        self.model_dir = os.path.dirname(__file__) + model_dir
        self.mean = dict(frames = 0, flow_x = 0, flow_y = 0)
        self.count = 0 

        print("Loading model")
        chkpt = torch.load(self.model_dir, map_location=lambda storage, loc: storage)
        self.model = CNNBlockFrameFlow()
        self.model.load_state_dict(chkpt["model"])
        if self.use_cuda == True:
            self.model.cuda()
        self.model.eval()      # BN and Dropout are different during training and inference    

        # Setup parameters for optical flow.
        self.flow_params = dict(winsize=20, iterations=1,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
        pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)


    def main(self):
        rospy.init_node('action_recogintion_node')
        self.sub_ = rospy.Subscriber('/csi_cam/image_raw/compressed', CompressedImage, self.callback)
        # self.sub_ = rospy.Subscriber('/csi_cam/image_raw', Image, self.callback)
        self.pub_ = rospy.Publisher('action_recogintion/camera_prediction', String, queue_size=10)
        rospy.spin()


    def cal_mean(self):
        if self.count < 20:
            self.mean['frames'] = ( self.mean['frames']*(self.count-1) + 
                                np.mean(self.current_block_image) ) / self.count
            self.mean['flow_x'] = ( self.mean['flow_x']*(self.count-1) + 
                                np.mean(self.current_block_flow_x[1:]) ) / self.count
            self.mean['flow_y'] = ( self.mean['flow_y']*(self.count-1) + 
                                np.mean(self.current_block_flow_y[1:]) ) / self.count
        else:
            self.mean = dict(frames = 0, flow_x = 0, flow_y = 0)
            self.count = 0


    def callback(self, data):

        # start = time.clock()       
        #################### Data In #####################
        try:
            self.br = CvBridge()
            # For /csi_cam/image_raw/compressed topic
            np_arr = np.fromstring(data.data, np.uint8)
            img = cv2.cvtColor( cv2.imdecode(np_arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY )

            # For /csi_cam/image_raw topic.
            # img = self.br.imgmsg_to_cv2(data,'mono8') 
        except CvBridgeError as e:
            print(e)
        # surprisingly, CompressedImage(~0.02s) processing is slower than that of RawImage(~0.002s)
        # print("cvbridge delay", time.clock()-start)
        
        img = img_as_ubyte(resize(img, (60,80), anti_aliasing=False))	# anti-aliasing off: ~0.02s. On: ~0.2s
        self.data.append( {"image":img} )
        self.prev_frame = img

        flow_x = np.zeros((30, 40), dtype=np.float32)
        flow_y = np.zeros((30, 40), dtype=np.float32)
       
        # Calculate optical flow
        if len(self.data) > 1:      
            flows = cv2.calcOpticalFlowFarneback(self.prev_frame, img, **self.flow_params)
            for r in range(30):
                for c in range(40):
                    flow_x[r, c] = flows[r*2, c*2, 0]
                    flow_y[r, c] = flows[r*2, c*2, 1]
        else:       
            pass    # The first frame does not have optical flow map

        self.data[-1]["flow_x"] = flow_x
        self.data[-1]["flow_y"] = flow_y

        #################### Data Out #####################
        if len(self.data) >= 15:
            current_frame = self.data.pop(0)
            self.current_block_image.append(current_frame["image"])
            self.current_block_flow_x.append(current_frame["flow_x"])
            self.current_block_flow_y.append(current_frame["flow_y"])
          
            # slide window size: 5; numbers of input frames into the CNN: 15, 14, 14
            if len(self.current_block_image) == 15:
                self.count += 1    
                self.cal_mean()
                score = getscore(self.model, self.current_block_image, self.current_block_flow_x[1:], 
                                            self.current_block_flow_y[1:], self.mean, self.use_cuda)    
                score -= np.max(score)      # deduct the maximum to avoid overflowing of exponent
                p = np.e**score / np.sum(np.e**score)
                pred = CATEGORIES[np.argmax(p)] 
                self.pub_.publish(pred)
                self.current_block_image = self.current_block_image[5:]
                self.current_block_flow_x = self.current_block_flow_x[5:]
                self.current_block_flow_y = self.current_block_flow_y[5:]
                print(pred)


# Numbe of frames in a block is 15. 
# A snippet consists of several blocks. A snippet generats one action recognition predicition.
# snippet_duration  =  frames_in_a_snippet/fps  =  15 * num_of_blocks_in_a_snippet / fps
if __name__ == "__main__":

    model_dir = rospy.get_param("/action_recognition/model_dir")
    use_cuda = rospy.get_param("/action_recognition/use_cuda")
    block_num = rospy.get_param("/action_recognition/block_num")

    action_recogintion(model_dir, use_cuda, block_num).main()

