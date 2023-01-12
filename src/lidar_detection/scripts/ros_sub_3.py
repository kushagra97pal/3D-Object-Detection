#!/usr/bin/env python3
import rospy
import math
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import argparse
import glob
from pathlib import Path
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from collections import defaultdict
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pcdet.datasets.processor.data_processor import DataProcessor
import time
import random
import cProfile
import pstats
import io
from pstats import SortKey
import ros_numpy
# from datetime import datetime as dt

# import open3d as o3d

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        print(self.sample_file_list)

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def marker_data(msg, count, class_type, class_scores):
    marker_est = Marker()
    marker_est.header.frame_id = "velodyne"
    marker_est.id = count
    marker_est.ns = "pred_" + str(marker_est.id)
    marker_est.type = Marker.CUBE
    marker_est.action = Marker.ADD
    marker_est.pose.orientation.w = msg[9]
    marker_est.pose.orientation.x = msg[6]
    marker_est.pose.orientation.y = msg[7]
    marker_est.pose.orientation.z = msg[8]
    marker_est.pose.position.x = msg[0]
    marker_est.pose.position.y = msg[1]
    marker_est.pose.position.z = msg[2]
    # marker_est.text = class_scores
    if class_type == 1:
        marker_est.color.r, marker_est.color.g, marker_est.color.b = (0, 255, 0)
    elif class_type == 2:
        marker_est.color.r, marker_est.color.g, marker_est.color.b = (255, 0, 0)
    elif class_type == 3:
        marker_est.color.r, marker_est.color.g, marker_est.color.b = (0, 0, 255)
    marker_est.color.a = 0.3
    marker_est.scale.x, marker_est.scale.y, marker_est.scale.z = (msg[3], msg[4], msg[5])

    marker_est_2 = Marker()
    marker_est_2.header.frame_id = "velodyne"
    marker_est_2.id = random.randint(500,5000)
    marker_est_2.ns = "pred_" + str(marker_est_2.id)
    marker_est_2.type = Marker.TEXT_VIEW_FACING
    marker_est_2.action = Marker.ADD
    marker_est_2.pose.orientation.w = msg[9]
    marker_est_2.pose.orientation.x = msg[6]
    marker_est_2.pose.orientation.y = msg[7]
    marker_est_2.pose.orientation.z = msg[8]
    marker_est_2.pose.position.x = msg[0]
    marker_est_2.pose.position.y = msg[1]
    marker_est_2.pose.position.z = msg[2]
    marker_est_2.text = str(class_scores)
    marker_est_2.color.r, marker_est_2.color.g, marker_est_2.color.b = (255, 255, 0)
    marker_est_2.color.a = 1
    marker_est_2.scale.x, marker_est_2.scale.y, marker_est_2.scale.z = (1, 1, 1)

    return marker_est, marker_est_2
    
def parse_config(cfg_file):
    parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
    #                     help='specify the config for demo')
    # parser.add_argument('--data_path', type=str, default='demo_data',
    #                     help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    # parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()
    cfg_from_yaml_file(cfg_file, cfg)

    return args, cfg

def collate_batch(batch_list, _unused=False):
    data_dict = defaultdict(list)
    for cur_sample in batch_list:
        for key, val in cur_sample.items():
            data_dict[key].append(val)
    batch_size = len(batch_list)
    ret = {}

    for key, val in data_dict.items():
        try:
            
            if key in ['points', 'voxel_coords']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            
            else:
                ret[key] = np.stack(val, axis=0)
        except:
            print('Error in collate_batch: key=%s' % key)
            raise TypeError

    ret['batch_size'] = batch_size
    return ret


def get_quaternion_from_euler(yaw):
    roll = 0
    pitch = 0

    """
    Convert an Euler angle to a quaternion.
    
    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.
    
    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
    return [qx, qy, qz, qw]

marker_all = MarkerArray()
count_list = []
frame_id_count = 0
count_box = 0
start = 0
end = 0

args, cfg = parse_config("/home/hyphen/iassd_catkin_ws/src/lidar_detection/scripts/IA-SSD/tools/cfgs/kitti_models/IA-SSD.yaml")
logger = common_utils.create_logger()
logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path("../data/kitti/test_folder"), logger=logger)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
model.cuda()
model.eval()
print("Model_loaded")

def get_xyzi_points(cloud_array, remove_nans=True, dtype=np.float32):
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
	a 3xN matrix.
    '''
    # remove crap points
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z']) & np.isfinite(cloud_array['intensity'])
        cloud_array = cloud_array[mask]
    
    # pull out x, y, z and i values
    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    # print(cloud_array['x'])
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    points[...,3] = cloud_array['intensity']

    return points

def callback(data):
    # ob = cProfile.Profile()
    # ob.enable()
    # full_time_start = time.time()
    global frame_id_count
    global count_box    
    global start, end

    points_32 = ros_numpy.point_cloud2.pointcloud2_to_array(data)
    points_32 = ros_numpy.point_cloud2.get_xyzi_points(points_32)
    
    points_32[:,3] = points_32[:,3]/256
    
    point_feature_encoder = PointFeatureEncoder(
            cfg.DATA_CONFIG.POINT_FEATURE_ENCODING,
            point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        )
    data_processor = DataProcessor(
        cfg.DATA_CONFIG.DATA_PROCESSOR, point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
        training=False, num_point_features=point_feature_encoder.num_point_features
    )
    
    with torch.no_grad():
        data_dict = {'points' : points_32, 'frame_id' : frame_id_count, 'use_lead_xyz' : True}
        data_dict = point_feature_encoder.forward(data_dict)
        data_dict = data_processor.forward(data_dict=data_dict)
        data_dict = collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        # time_start = time.time()
        pred_dicts, _ = model.forward(data_dict)
        # time_end = time.time()
        # print("time_diff: ", time_end - time_start)
        ref_boxes = pred_dicts[0]['pred_boxes']
        ref_labels = pred_dicts[0]['pred_labels']
        ref_scores = pred_dicts[0]['pred_scores']

        if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
            ref_boxes = ref_boxes.detach().cpu().numpy()
            ref_labels = ref_labels.detach().cpu().numpy()
            ref_scores = ref_scores.detach().cpu().numpy()
        
        boxes3d = ref_boxes
        class_labels = ref_labels
        class_scores = ref_scores
        bbox_coord_all = []

        box_points = np.zeros(len(points_32), dtype=[('x', np.float32),('y', np.float32),('z', np.float32),('intensity', np.float32)])
        box_points['x'] = points_32[:,0]
        box_points['y'] = points_32[:,1]
        box_points['z'] = points_32[:,2]
        box_points['intensity'] = points_32[:,3]

        for item in range(boxes3d.shape[0]):

            length_range = (boxes3d[item][0] - (boxes3d[item][3]/2 + 0.75), boxes3d[item][0] + (boxes3d[item][3]/2 + 0.75))
            breadth_range = (boxes3d[item][1] - (boxes3d[item][4]/2 + 0.75) , boxes3d[item][1] + (boxes3d[item][4]/2 + 0.75))
            height_range = (boxes3d[item][2] - (boxes3d[item][5]/2 + 0.75), boxes3d[item][2] + (boxes3d[item][5]/2 + 0.75))

            x_points = points_32[:, 0]
            y_points = points_32[:, 1]
            z_points = points_32[:, 2]

            x_filt = np.logical_and((x_points > length_range[0]), (x_points < length_range[1]))
            y_filt = np.logical_and((y_points > breadth_range[0]), (y_points < breadth_range[1]))
            z_filt = np.logical_and((z_points > height_range[0]), (z_points < height_range[1]))

            myFilter = np.logical_and(x_filt, y_filt)   
            myFilter2 = np.logical_and(myFilter, z_filt)
            indices = np.argwhere(myFilter2).flatten()

            box_points['x'][indices] = 0
            box_points['y'][indices] = 0
            box_points['z'][indices] = 0
            
            count_box += 1
            quart = get_quaternion_from_euler(boxes3d[item][6])
            bbox_coord = boxes3d[item][:6]
            bbox_coord = np.concatenate((bbox_coord, quart))
            cube_marker, score_marker = marker_data(bbox_coord, count_box, class_labels[item], class_scores[item])
            marker_all.markers.append(cube_marker)
            marker_all.markers.append(score_marker)
            bbox_coord_all.append(bbox_coord)
        

    points_32_array = ros_numpy.point_cloud2.array_to_pointcloud2(box_points)
    points_32_array.header.frame_id = 'velodyne'
    pub_1.publish(points_32_array)
        
    frame_id_count += 1  

    end = 2*(count_box - ref_boxes.shape[0])

    if frame_id_count>1:

        for uniq_marker in range(start,end):
            marker_all.markers[uniq_marker].action = Marker.DELETE
        start = end
        pub.publish(marker_all) 
        
    else:
        pub.publish(marker_all) 
    # full_time_end = time.time()
    # print("callback_time: ", full_time_end - full_time_start)
    # ob.disable()
    # sec = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(ob, stream=sec).sort_stats(sortby)
    # ps.print_stats()    
    # print(sec.getvalue())

def listener():
    rospy.init_node('ros_sub', anonymous=True)
    rospy.Subscriber("/lidar103/velodyne_points", PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=1)
    pub_1 = rospy.Publisher("/velodyne_points", PointCloud2, queue_size=1)


    listener()
