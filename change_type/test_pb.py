# -*- coding: utf-8 -*-
# @Time : 11/03/20 下午 05:13
# @Author : yuhao
# @File : test_pb.py
# @Software: PyCharm
# @Details:
import cv2
import numpy as np
from utils import utils
import tensorflow as tf
from utils.nms_utils import gpu_nms
from model import yolov3
from utils.misc_utils import parse_anchors
from utils.plot_utils import plot_one_box,get_color_table


return_elements = ["input_data:0", 'yolov3/yolov3_head/feature_map_1:0', 'yolov3/yolov3_head/feature_map_2:0', 'yolov3/yolov3_head/feature_map_3:0']
pb_file         = "./yolov3.pb"
image_path      = ""
num_classes     = 1
input_size      = 416
graph           = tf.Graph()

original_image_ = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image_, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
height_ori, width_ori = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
print(return_tensors[0])

with tf.Session(graph=graph) as sess:
    feature_map_1, feature_map_2, feature_map_3 = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={ return_tensors[0]: image_data})


    feature_map_1 = tf.identity(feature_map_1, name='feature_map_3')
    feature_map_2 = tf.identity(feature_map_2, name='feature_map_3')
    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

    yolo_model = yolov3(1, parse_anchors('./data/yolo_anchors.txt'))
    pred_boxes, pred_confs, pred_probs = yolo_model.predict([feature_map_1, feature_map_2, feature_map_3])
    pred_scores = pred_confs * pred_probs
    boxes_, scores_, labels_ = gpu_nms(pred_boxes, pred_scores, 1, max_boxes=200, score_thresh=0.5,
                                        nms_thresh=0.45)

    boxes_ = boxes_.eval()
    scores_ = scores_.eval()
    labels_ = labels_.eval()

    boxes_[:, 0] *= (width_ori / float(416))
    boxes_[:, 2] *= (width_ori / float(416))
    boxes_[:, 1] *= (height_ori / float(416))
    boxes_[:, 3] *= (height_ori / float(416))


    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(original_image_, [x0, y0, x1, y1], label='person', color=(255,255,255))
    cv2.imwrite('detection_result.jpg', original_image_)
