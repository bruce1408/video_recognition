# -*- coding: utf-8 -*-
# @Time : 21/01/20 下午 07:57
# @Author : yuhao
# @File : ckpt2pb_2.py
# @Software: PyCharm
# @Details:
import tensorflow as tf
from tensorflow import saved_model as sm
from model import yolov3
from utils.misc_utils import parse_anchors
from utils.nms_utils import gpu_nms




with tf.Session() as sess:

    input_data = tf.placeholder(tf.float32, [1, 416, 416, 3], name='input_data')
    yolo_model = yolov3(1, parse_anchors('./data/yolo_anchors.txt'))
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, 1, max_boxes=200, score_thresh=0.5,
                                    nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, './my_checkpoint4/model-epoch_380_step_83438_loss_3.9319_lr_1e-05')


    builder = sm.builder.SavedModelBuilder('./weights5')

    X_TensorInfo = sm.utils.build_tensor_info(input_data)





    y_TensorInfo1 = sm.utils.build_tensor_info(boxes)
    y_TensorInfo2 = sm.utils.build_tensor_info(scores)
    y_TensorInfo3 = sm.utils.build_tensor_info(labels)
    SignatureDef = sm.signature_def_utils.build_signature_def(
        inputs={'input': X_TensorInfo},
        outputs={'output_1': y_TensorInfo1, 'output_2': y_TensorInfo2, 'output_3': y_TensorInfo3},
        method_name='what'
    )
    builder.add_meta_graph_and_variables(sess, tags=[sm.tag_constants.TRAINING],
                                         signature_def_map={sm.signature_constants.CLASSIFY_INPUTS: SignatureDef}
                                         )

builder.save()











