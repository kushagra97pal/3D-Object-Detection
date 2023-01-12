#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
# from vision_msgs.msg import d
import yaml

def callback(data):
    # print(data.fields)
    # print(len(data.fields))
    points = point_cloud2.read_points(data)
    # len(points)
    np_x = []
    np_y = []
    np_z = []
    np_i = []

    for p in points:
        np_x.append(p[0])
        np_y.append(p[1])
        np_z.append(p[2])
        np_i.append(p[3]/256)
    
    np_o = np.zeros(len(np_x), dtype='float32')
    np_x = np.asarray(np_x, dtype='float32')
    np_y = np.asarray(np_y, dtype='float32')
    np_z = np.asarray(np_z, dtype='float32')
    np_i = np.asarray(np_i, dtype='float32')
    
    
    points_32 = np.transpose(np.vstack((np_o, np_x, np_y, np_z, np_i)))
    print(points_32.shape)

    # cfg_file = "cfgs/kitti_models/IA-SSD.yaml"
    # weight_file = "IA-SSD.pth"
    # data_path = False
    # extension = False
    



def listener():
    rospy.init_node('ros_sub', anonymous=True)
    rospy.Subscriber("/lidar103/velodyne_points", PointCloud2, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()