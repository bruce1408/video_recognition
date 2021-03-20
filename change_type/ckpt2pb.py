# -*- coding: utf-8 -*-
# @Time : 11/03/20 下午 04:55
# @Author : yuhao
# @File : ckpt2pb.py
# @Software: PyCharm
# @Details:
import tensorflow as tf
from model import yolov3
from utils.misc_utils import parse_anchors



with tf.Session() as sess:
    output_node_names = ["input_data", 'yolov3/yolov3_head/feature_map_1', 'yolov3/yolov3_head/feature_map_2', 'yolov3/yolov3_head/feature_map_3']
    input_data = tf.placeholder(tf.float32, [1, 416, 416, 3], name='input_data')
    yolo_model = yolov3(1, parse_anchors('./data/yolo_anchors.txt'))
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    saver = tf.train.Saver()
    saver.restore(sess, './my_checkpoint4/model-epoch_380_step_83438_loss_3.9319_lr_1e-05')


    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                       input_graph_def=sess.graph.as_graph_def(),
                                                                       output_node_names=output_node_names)

    with tf.gfile.GFile('./yolov3.pb', "wb") as f:
        f.write(converted_graph_def.SerializeToString())















































'''

import tensorflow as tf
from model import yolov3

pb_file = "./yolov3_coco.pb"
ckpt_file = "./checkpoint/yolov3_coco_demo.ckpt"
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, name='input_data')

model = yolov3(input_data, trainable=False)


sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())




'''