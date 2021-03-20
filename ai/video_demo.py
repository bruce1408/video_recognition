# human detecting
from __future__ import division, print_function
import cv2, time, datetime, json
import gc
import os
import numpy as np
import tensorflow as tf
from ai.core import utils
from ai import config as cfg
from tools.mqtt_tool import MqttTool
from multiprocessing import Process, Manager, Queue
import signal
from tensorflow import saved_model as sm
from ai import move_recognition as m_r

# # slowfast
# from time import time #maodun?
# import pandas as pd
# import torch
#
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog
#
# import slowfast.utils.checkpoint as cu
# import slowfast.utils.distributed as du
# from slowfast.utils import logging
# from slowfast.utils import misc
# from slowfast.datasets import cv2_transform
# from slowfast.models import model_builder
# from slowfast.datasets.cv2_transform import scale
#
# logger = logging.get_logger(__name__)
# np.random.seed(20)

## begin
s = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
graph = tf.Graph()
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
path = cfg.path
anchors = cfg.anchors
coco = cfg.coco
anchors = utils.parse_anchors(anchors)
classes = utils.read_class_name(coco)  # 标签只有一类为{0: 'person'}
input_size = 416
num_class = 1
color_table = utils.get_color_table(1)


def run_sess(img, sess, SignatureDef):
    X_TensorInfo = SignatureDef.inputs['input']
    y_TensorInfo_1 = SignatureDef.outputs['output_1']
    y_TensorInfo_2 = SignatureDef.outputs['output_2']
    y_TensorInfo_3 = SignatureDef.outputs['output_3']
    # 解析得到具体 Tensor
    # .get_tensor_from_tensor_info() 函数中可以不传入 graph 参数，TensorFlow 自动使用默认图
    X = sm.utils.get_tensor_from_tensor_info(X_TensorInfo, sess.graph)
    y1 = sm.utils.get_tensor_from_tensor_info(y_TensorInfo_1, sess.graph)
    y2 = sm.utils.get_tensor_from_tensor_info(y_TensorInfo_2, sess.graph)
    y3 = sm.utils.get_tensor_from_tensor_info(y_TensorInfo_3, sess.graph)
    boxes_, scores_, labels_ = sess.run([y1, y2, y3], feed_dict={X: img})

    return boxes_, scores_, labels_


def video_obj(frame, sess, read, ca_id, SignatureDef, dic_camera, q_frame_all, frame_all_dict, fps):
    """
    计算工位状态,
    :param frame:
    :param sess:
    :param read:
    :param ca_id:
    :param SignatureDef:
    :param dic_camera:
    :param q_frame_all:
    :param frame_all_dict:
    :param fps:
    :return:
    """
    # TODO
    if True:
        img, resize_ratio, dw, dh = utils.letterbox_resize(frame, input_size, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.0
    box_dic = dict()
    # after resize, make a predicting with tensorflow, detect person or not
    box_dic['resize_ratio'] = resize_ratio
    box_dic['dw'] = dw
    box_dic['dh'] = dh
    box_dic['classes'] = classes
    box_dic['color_table'] = color_table
    # print('class is:', classes)
    # print('color-table is:', color_table)
    # 模型预测信息
    boxes_, scores_, labels_ = run_sess(img, sess, SignatureDef)  # 开始预测模型返回框,得分,标签
    box_dic['boxes_'] = boxes_
    box_dic['scores_'] = scores_
    box_dic['labels_'] = labels_
    # print('the boxes is:', boxes_)
    # print('the scores is:', scores_)
    # print('the labels is:', labels_)

    predict_box_list = utils.predict_box(frame, box_dic)  # 预测坐标
    # print('predict box list is:', predict_box_list)
    # 获取实际坐标 true_box_list=》[(工位id, (min_x, min_y, max_x, max_y) )]    dutyid_list -> 工位id
    true_box_list, dutyid_list = utils.true_box(read, frame, ca_id, dic_camera)
    # print('the true box list , duy id is:', true_box_list, dutyid_list)
    res_list = utils.compute_status(true_box_list, predict_box_list, dutyid_list, ca_id)
    # print('res_list is:', res_list)
    # 这里是逐帧保存数据，目前是存在一个字典里面    结构{camera_id：[frame, frame, frame, frame...]}
    # del_list = []
    # for c_data in range(len(q_frame_all)):
    #     if q_frame_all[c_data][0] == ca_id:
    #         del_list.append(c_data)
    #         c_frame = q_frame_all[c_data][1]
    #         # _ = utils.predict_box(c_frame, box_dic)
    #         for p in predict_box_list:
    #             utils.plot_one_box(c_frame, [p[1][0], p[1][1], p[1][2], p[1][3]],
    #                          label='person',
    #                          color=(255, 255, 255))
    #
    #
    #         _, _ = utils.true_box(read, c_frame, ca_id, dic_camera)
    #         if ca_id not in frame_all_dict.keys():
    #             frame_all_dict[ca_id] = [c_frame]
    #         else:
    #             frame_all_dict[ca_id].append(c_frame)
    #         if len(del_list) == fps:
    #             break

    # del_list.sort(reverse=True)
    # for i in del_list:
    #     del q_frame_all[i]

    # # 生成视频
    # if len(q_frame_all) <= fps*len(frame_all_dict.keys()):
    #     for key in frame_all_dict.keys():
    #         video_writer = cv2.VideoWriter('./res1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30,
    #                                        (frame_all_dict[key][0].shape[1], frame_all_dict[key][0].shape[0]))
    #         for item in frame_all_dict[key]:
    #             video_writer.write(item)
    #         video_writer.release()
    print("res_list is:", res_list)
    return res_list


def view_video(frame, ca_id, start_time):
    end_time = time.time()
    res_img = frame
    cv2.putText(res_img, 'cost: {:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,
                fontScale=1, color=(0, 0, 255), thickness=5)

    cv2.namedWindow(ca_id, cv2.WINDOW_NORMAL)
    cv2.imshow(ca_id, res_img)
    cv2.waitKey(1)


def end_t_json(dic, mqtt_tool, am_home_cut, pm_home_cut, dic_camera, ca_id, frame, c_time, offduty_limit, mask,
               config_dict, q_frame_difference, fps):
    """
    脱岗离岗
    dic:每个工位的状态(刚在视频帧上预测得到的，实时)
    am_home_cut:上午早退阀值
    pm_home_cut:下午早退阀值
    dic_camera:每个工位的状态(自己定义的变量存储工位状态)
    offduty_limit:检测不在岗状态多长时间为脱岗
    c_time:当前时间(并非实时，而是计算各个指标时的时间)
    frame:图像
    """
    ki = ca_id
    if dic.get(ca_id, -1) != -1:
        res_dic = dict()
        res_data = []
        for kj in dic[ki]:

            st_time = datetime.datetime.now()
            st_time = st_time.strftime('%H:%M:%S')

            outduty = False
            if dic[ki][kj]['onduty'] == 0 and dic[ki][kj]['offduty'] == 0:
                continue
            if dic[ki][kj]['onduty'] < int(1):
                outduty = True
                res_dic["dutyid"] = kj
                res_dic["status"] = "offduty"
                res_dic["sleep"] = "no"
                dic_camera[ki][kj]['on_off_duty'] = "offduty"
                dic_camera[ki][kj]['sleep_start_time'] = st_time
                dic_camera[ki][kj]['sleepduty'] = False
            else:
                outduty = False
                res_dic["dutyid"] = kj
                res_dic["status"] = "onduty"
                dic_camera[ki][kj]['on_off_duty'] = "onduty"

                flag_mask, sleepduty_limit, q_frame_difference = utils.compute_sleep(ki, kj, mask, config_dict,
                                                                                     q_frame_difference)
                print(ki + kj)
                print(q_frame_difference[ki + kj])
                print(flag_mask)
                if len(q_frame_difference[ki + kj]) > int(sleepduty_limit * 60 * 30 / fps):
                    del q_frame_difference[ki + kj][int(sleepduty_limit * 60 * 30 / fps):]
                if len(q_frame_difference[ki + kj]) >= int(sleepduty_limit * 60 * 30 / fps) - 1:
                    sum_num = 0
                    for difference in q_frame_difference[ki + kj]:
                        if difference > 20:  # 这里认为帧差超过20像素为有相对位移
                            sum_num += 1
                    # 若sleepduty_limit分钟内检测的帧数中帧差超过20的数量小于检测总帧数的0.2则认为为睡觉状态
                    if sum_num < int(
                            sleepduty_limit * 60 * 30 / fps * 0.2):
                        flag_mask = False

                if dic_camera[ki][kj]['sleep_start_time'] is None:
                    dic_camera[ki][kj]['sleep_start_time'] = st_time

                if flag_mask:
                    dic_camera[ki][kj]['sleep_start_time'] = st_time
                    dic_camera[ki][kj]['sleepduty'] = False
                    res_dic["sleep"] = "no"
                else:
                    sleep_time = utils.time_cha(dic_camera[ki][kj]['sleep_start_time'])
                    # 会在睡岗超过sleepduty_limit后警报999999s，然后初始化状态
                    if sleepduty_limit * 60 <= sleep_time <= sleepduty_limit * 60 + 999999:
                        dic_camera[ki][kj]['sleepduty'] = True
                        res_dic["sleep"] = "yes"
                    elif sleep_time > sleepduty_limit * 60 + 999999:  # 警报超过999999s，初始化状态
                        dic_camera[ki][kj]['sleep_start_time'] = st_time
                        dic_camera[ki][kj]['sleepduty'] = False
                        res_dic["sleep"] = "no"

                if 'sleep' not in res_dic.keys():
                    res_dic["sleep"] = 'no'

            if not outduty:
                dic_camera[ki][kj]['starttime'] = st_time

            altertime = utils.time_cha(dic_camera[ki][kj]['starttime'])  # altertime：脱岗离岗时间

            if c_time < am_home_cut - 10 or c_time < pm_home_cut - 10:  # 如果再工作时间内
                if altertime > float(offduty_limit) * 60:  # TODO 实际生产中把 2 替换为 -> float(offduty_limit)*60 - 20

                    if not dic_camera[ki][kj]['alter']:
                        dic_camera[ki][kj]['altertime'] = altertime
                        dic_camera[ki][kj]['alter'] = True
                        res_dic["status"] = "offduty"
                        dic_camera[ki][kj]['on_off_duty'] = "offduty"
                    # if utils.time_cha(dic['altertime']) <= 10: # 10s
                    if altertime > dic_camera[ki][kj]['altertime'] + 999999:  # 999999s 脱岗离岗时间超过阈值999999s，警报持续999999s
                        st_time = datetime.datetime.now()
                        st_time = st_time.strftime('%H:%M:%S')
                        dic_camera[ki][kj]['starttime'] = st_time
                        dic_camera[ki][kj]['alter'] = False
                        res_dic["status"] = "onduty"
                        dic_camera[ki][kj]['on_off_duty'] = "onduty"
                else:
                    dic_camera[ki][kj]['alter'] = False
                    res_dic["status"] = "onduty"
                    dic_camera[ki][kj]['on_off_duty'] = "onduty"

                if res_dic != {}:
                    res_data.append(res_dic)
                    res_dic = dict()

        if len(res_data) != 0:
            x, y = frame.shape[0:2]
            res_img = cv2.resize(frame, (int(y / 3), int(x / 3)))
            base64_img = utils.cv2_base64(res_img)
            dic_json = {
                "cameraid": ki,
                "timestemp": time.time(),
                # "base64img": "data:image/jpeg;base64,{0}".format(str(base64_img, "utf-8")), # 图片解码
                "base64img": 'base64',
                "data": res_data}

            if len(dic_json) != 0:
                # mqtt_tool.pubish(json.dumps(dic_json))
                print('this is json >>>: ', dic_json)

            # TODO test start

            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cost_time = 0  # cost_time = (time.time() - lhy_start_time) * 1000
            cv2.putText(res_img, "camera_id:{0},timestemp:{1},cost:{2}".
                        format(ki, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                               cost_time), (25, 25), font, 1, (0, 255, 0), thickness=2, lineType=2)

            for index, resdata in enumerate(res_data):
                cv2.putText(res_img, json.dumps(resdata), (25, 25 * (index + 2)), font, 1, (0, 255, 0), thickness=2,
                            lineType=2)

            # cv2.imwrite("/home/yuhao/Desktop/model_read_json/video_surveillance/ai/res/end_t_{0}.jpg".
            # format(time.time() * 1000), res_img)


def end_c_json(dic, mqtt_tool, am_work, am_home, pm_work, pm_home,
               am_work_add, pm_work_add, am_home_cut, pm_home_cut, dic_camera, ca_id, frame, c_time, offduty_limit,
               mask, config_dict, q_frame_difference, fps):
    """
    迟到早退
    dic:每个工位的状态(刚在视频帧上预测得到的，实时)
    am_work:上午上班时间
    am_home:上午下班时间
    pm_work:下午上班时间
    pm_home:下午下班时间
    am_work_add:上午迟到阀值
    pm_work_add:下午迟到阀值
    am_home_cut:上午早退阀值
    pm_home_cut:下午早退阀值
    dic_camera:每个工位的状态(自己定义的变量存储工位状态)
    offduty_limit:检测不在岗状态多长时间为脱岗
    c_time:当前时间(并非实时，而是计算各个指标时的时间)
    frame:图像
    :return:
    """
    ki = ca_id
    if dic.get(ca_id, -1) != -1:
        res_data = []
        for kj in dic[ki]:
            # res_dic["dutyid"] = kj
            st_time = datetime.datetime.now()
            st_time = st_time.strftime('%H:%M:%S')

            res_dic = dict()
            if dic[ki][kj]['onduty'] == 0 and dic[ki][kj]['offduty'] == 0:
                continue
            if dic[ki][kj]['onduty'] < int(1):
                res_dic["dutyid"] = kj
                res_dic["status"] = "offduty"
                res_dic["sleep"] = "no"
                dic_camera[ki][kj]['sleep_start_time'] = st_time
                dic_camera[ki][kj]['sleepduty'] = False
                dic_camera[ki][kj]['on_off_duty'] = "offduty"
            else:
                res_dic["dutyid"] = kj
                res_dic["status"] = "onduty"
                sleepduty_limit, flag_mask, q_frame_difference = utils.compute_sleep(ki, kj, mask, config_dict,
                                                                                     q_frame_difference)

                if len(q_frame_difference[ki][kj]) > int(sleepduty_limit * 60 * 30 / fps):
                    del q_frame_difference[ki][kj][int(sleepduty_limit * 60 * 30 / fps):]
                if len(q_frame_difference[ki][kj]) >= int(sleepduty_limit * 60 * 30 / fps):
                    sum_num = 0
                    for difference in q_frame_difference[ki][kj]:
                        if difference > 20:  # 这里认为帧差超过20像素为有相对位移
                            sum_num += 1
                    if sum_num < int(
                            sleepduty_limit * 60 * 30 / fps * 0.2):  # 若sleepduty_limit分钟内检测的帧数中帧差超过20的数量小于检测总帧数的0.2则认为为睡觉状态
                        flag_mask = False

                if dic_camera[ki][kj]['sleep_start_time'] is None:
                    dic_camera[ki][kj]['sleep_start_time'] = st_time

                if flag_mask:
                    dic_camera[ki][kj]['sleep_start_time'] = st_time
                    dic_camera[ki][kj]['sleepduty'] = False
                    res_dic["sleep"] = "no"
                else:
                    sleep_time = utils.time_cha(dic_camera[ki][kj]['sleep_start_time'])
                    if sleepduty_limit * 60 <= sleep_time <= sleepduty_limit * 60 + 999999:  # 会在睡岗超过sleepduty_limit后警报999999s，然后初始化状态
                        dic_camera[ki][kj]['sleepduty'] = True
                        res_dic["sleep"] = "yes"
                    elif sleep_time > sleepduty_limit * 60 + 999999:  # 警报超过999999s，初始化状态
                        dic_camera[ki][kj]['sleep_start_time'] = st_time
                        dic_camera[ki][kj]['sleepduty'] = False
                        res_dic["sleep"] = "no"

            if dic[ki][kj]['onduty'] == 0 and dic[ki][kj]['offduty'] == 0:
                continue
            if am_work <= c_time < am_work_add or pm_work <= c_time < pm_work_add:

                if dic[ki][kj]['onduty'] >= int(1):  # 当大于该阈值时认为是在岗
                    dic_camera[ki][kj]['abnormal'] = False
                st_time = datetime.datetime.now()
                st_time = st_time.strftime('%H:%M:%S')

                if not dic_camera[ki][kj]['abnormal']:
                    dic_camera[ki][kj]['starttime'] = st_time

                altertime = utils.time_cha(dic_camera[ki][kj]['starttime'])
                if c_time > am_work_add - 10 or c_time > pm_work_add - 10:
                    if altertime > float(offduty_limit) * 60:  # TODO 实际生产中把 2 替换为->float(offduty_limit)*60 - 20
                        res_dic["status"] = "gowork_late"
                        dic_camera[ki][kj]['on_off_duty'] = "gowork_late"
                        # res_dic["prob_iou"] = 0

            elif am_home_cut <= c_time < am_home or pm_home_cut <= c_time < pm_home:
                if dic[ki][kj]['onduty'] >= int(1):  # 当大于该阈值时认为是在岗
                    dic_camera[ki][kj]['abnormal'] = False

                st_time = datetime.datetime.now()
                st_time = st_time.strftime('%H:%M:%S')
                if not dic_camera[ki][kj]['abnormal']:
                    dic_camera[ki][kj]['starttime'] = st_time
                altertime = utils.time_cha(dic_camera[ki][kj]['starttime'])
                if c_time > am_home - 10 or c_time > pm_home - 10:
                    if altertime > float(offduty_limit) * 60:  # TODO 实际生产中把 2 替换为->float(offduty_limit)*60 - 20
                        res_dic["status"] = "gohome_early"
                        dic_camera[ki][kj]['on_off_duty'] = "gohome_early"
            res_data.append(res_dic)
        if len(res_data) != 0:
            x, y = frame.shape[0:2]
            res_img = cv2.resize(frame, (int(y / 3), int(x / 3)))
            base64_img = utils.cv2_base64(res_img)
            dic_json = {
                "cameraid": ki,
                "timestemp": time.time(),
                "base64img": 'base64_img',
                "data": res_data}

            if len(dic_json) != 0:
                # mqtt_tool.pubish(json.dumps(dic_json))
                print('this is json >>>: ', dic_json)

            # TODO start
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cost_time = 0  # cost_time = (time.time() - lhy_start_time) * 1000
            cv2.putText(res_img, "camera_id:{0},timestemp:{1},cost:{2}".format(ki, time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                                 time.localtime()),
                                                                               cost_time), (25, 25), font, 1,
                        (0, 255, 0), thickness=2, lineType=2)

            for index, resdata in enumerate(res_data):
                cv2.putText(res_img, json.dumps(resdata), (25, 25 * (index + 2)), font, 1, (0, 255, 0), thickness=2,
                            lineType=2)

            # cv2.imwrite("/home/yuhao/Desktop/model_read_json/video_surveillance/ai/res/end_t_{0}.jpg".format(time.time() * 1000), res_img)
            mqtt_tool.pubish(json.dumps(dic_json))
            # print('this is json >>>: ', dic_json)
            # TODO end


def r_human(mqtt_tool, detectFrameList, queue, frameDiff, fps, frameWholeData, human_exists):
    """
    检测人头部分代码 调用代码 video_obj
    :param mqtt_tool:
    :param detectFrameList: 保存的是视频id, 间隔帧数据和帧差的一个list
    :param queue: 共享队列
    :param frameDiff: 视频帧差
    :param fps: 帧数
    :param frameWholeData: 视频所有帧数据,视频id, 视频连续帧, 视频
    :param human_exists: 是否存在人头
    :return:
    """
    del human_exists[:]  # clear first, or will be added!
    with tf.Session(graph=graph, config=config) as sess:
        """
        tags 参数是 " train"
        """
        MetaGraphDef = sm.loader.load(sess, tags=[sm.tag_constants.TRAINING], export_dir=path)
        # 解析得到 SignatureDef protobuf
        SignatureDef_d = MetaGraphDef.signature_def
        SignatureDef = SignatureDef_d[sm.signature_constants.CLASSIFY_INPUTS]

        q_frame_difference = dict()  # 存放帧差数组
        frame_all_dict = dict()  # 存放所有帧(为了合成视频流)

        video_info, read = utils.read_config(cfg)
        print(cfg.config_json)
        # print('the video_info is : ', video_info)  # video id 号
        # print('the read info is: ', read)
        dic_state = dict()
        dic_state['checkout'] = False  # 是否离岗
        dic_state['checklate'] = False  # 是否迟到

        dic_camera = dict()  # 该视频id号的位置坐标信息
        dic_p = dict()
        dic_p['starttime'] = None
        dic_p['alter'] = False
        dic_p['altertime'] = None
        dic_p['abnormal'] = True
        dic_p['sleep_start_time'] = None
        dic_p['sleepduty'] = False
        dic_p['on_off_duty'] = 'onuty'

        # sdf 摄像头位置信息初始化
        for infos in read['camera_data']:
            camid = infos['cameraid']
            # 每一个工位copy一个dic_p
            dic_camera[camid] = {duty['dutyid']: dic_p.copy() for duty in infos['duty_data']}
            # print('the dic_camera is: ', dic_camera[camid])
            # print('the duty_data is: ', infos['duty_data'])
        # print('the dic_camera is: ', dic_camera[camid])
        # print('the duty_data is: ', infos['duty_data'])
        start_time = time.time()
        s_time = datetime.datetime.now()
        s_time = s_time.strftime('%H:%M:%S')

        duty_status = []
        queue.put(os.getpid())  # 将消费进程r的id喂给队列，由生产进程writeQueue来获取

        count = 0
        # detectFrameList 视频id, 间隔帧数据和帧差的一个list
        while True:
            if len(detectFrameList) > 0:
                try:
                    vid_lis = detectFrameList.pop()
                    # continue
                    frameDiff.insert(0, vid_lis)
                except:
                    continue
                if len(vid_lis) != 0:
                # if vid_lis != []:
                    camera_data = read['camera_data']
                    for ca_id, frame, mask_item in vid_lis:  # 从上面的detectFrameList 表里面拿出一个进行测试
                        # print(ca_id)
                        # flag等于True时处理脱岗离岗，等于False时处理迟到早退
                        flag = None
                        duty_data = None
                        for cam in camera_data:
                            if cam['cameraid'] == ca_id:
                                duty_data = cam['duty_data']  # 找到视频中位置坐标信息列表
                        for duty_data_dic in duty_data:  # 开始遍历视频位置坐标, 检查视频中每个岗位的工作状态
                            dutyid = duty_data_dic['dutyid']  # 工位id
                            offduty_limit = duty_data_dic['offduty_limit']  # 检测不在岗状态多长时间为脱岗
                            am_gowork_time = duty_data_dic['am_gowork_time']  # 上午上班时间
                            am_gohome_time = duty_data_dic['am_gohome_time']  # 上午下班时间
                            pm_gowork_time = duty_data_dic['pm_gowork_time']  # 下午上班时间
                            pm_gohome_time = duty_data_dic['pm_gohome_time']  # 下午下班时间
                            gowork_time_after = duty_data_dic['gowork_time_after']  # 上班后检测多久
                            gohome_time_before = duty_data_dic['gohome_time_before']  # 下班前多久检测
                            sleepduty_limit = duty_data_dic['sleepduty_limit']  # 超过多久算睡觉
                            playmobile_limit = duty_data_dic['playmobile_limit']  # 超过多久算玩手机

                            current_time = int(time.time())  # 当前时间
                            time_date = time.strftime("%Y%m%d")  # 当前日期年月日
                            # 上午上班时间, 时间戳的形式
                            am_work = int(time.mktime(
                                time.strptime(time_date + ''.join(str(am_gowork_time).split(':')), "%Y%m%d%H%M%S")))
                            # 上午迟到阀值的时间戳,也就是上班之后的10分钟.
                            am_work_add = am_work + int(gowork_time_after) * 60
                            # 上午下班时间
                            am_home = int(time.mktime(
                                time.strptime(time_date + ''.join(str(am_gohome_time).split(':')), "%Y%m%d%H%M%S")))
                            # 上午早退阀值
                            am_home_cut = am_home - int(gohome_time_before) * 60
                            # 下午上班时间
                            pm_work = int(time.mktime(
                                time.strptime(time_date + ''.join(str(pm_gowork_time).split(':')), "%Y%m%d%H%M%S")))
                            # 下午迟到阀值
                            pm_work_add = pm_work + int(gowork_time_after) * 60
                            # 下午下班时间
                            pm_home = int(time.mktime(
                                time.strptime(time_date + ''.join(str(pm_gohome_time).split(':')), "%Y%m%d%H%M%S")))
                            # 下午早退阀值
                            pm_home_cut = pm_home - int(gohome_time_before) * 60
                            # 脱岗离岗
                            if am_work_add <= current_time < am_home_cut or pm_work_add <= current_time < pm_home_cut:
                                flag = True
                            # 迟到早退
                            elif am_work <= current_time < am_work_add or abs(am_home_cut) <= current_time < am_home or \
                                    pm_work <= current_time < pm_work_add or abs(pm_home_cut) <= current_time < pm_home:
                                flag = False
                        # 如果没有脱岗离岗和早退就设置为false, 那么就不去检测
                        if flag is None:
                            dic_state['checkout'] = False
                            dic_state['checklate'] = False
                            print('不在检测时间范围内')
                            time.sleep(10)
                            continue

                        res_list = video_obj(frame, sess, read, ca_id, SignatureDef, dic_camera, frameWholeData,
                                             frame_all_dict, fps)  # 计算的状态
                        duty_status.extend(res_list)
                        human_exists.extend(res_list)
                        l_time = utils.time_cha(s_time)  # 计算时间差
                        # break #for ca_id, frame, mask_item in vid_lis:
                    break  # while True:
            # 已经得到每个工位的状态
            # if int(l_time) >= 0:  ###### 2秒汇总一次(可以调节，这里指的是每隔几秒统计一次结果)
            #     count += 1
            #     # print(count)
            #     s_time = datetime.datetime.now()
            #     s_time = s_time.strftime('%H:%M:%S')  # 重新初始化为当前时间
            #     dic = utils.compute_duty(duty_status)  ####  duty_status这里是 两秒内所有帧/fps 帧数和，若这些帧中某一个工位只要检测到一次人就证明此人在岗
            #     if flag is True:   ##脱岗离岗
            #         if not dic_state['checkout']:
            #             t_time_int = datetime.datetime.now()
            #             t_time_int = t_time_int.strftime('%H:%M:%S')
            #             for key, value in dic_camera.items():
            #                 for pi, pj in value.items():
            #                     pj['abnormal'] = True
            #                     pj['alter'] = False
            #                     pj['starttime'] = t_time_int
            #
            #             dic_state['checkout'] = True
            #             dic_state['checklate'] = False
            #         end_t_json(dic, mqtt_tool, am_home_cut, pm_home_cut, dic_camera, ca_id, frame, current_time, offduty_limit, mask_item, read, q_frame_difference, fps)
            #     elif flag is False:   ##迟到早退
            #         if not dic_state['checklate']:
            #             c_time_int = datetime.datetime.now()
            #             c_time_int = c_time_int.strftime('%H:%M:%S')
            #             for key, value in dic_camera.items():
            #                 for pi, pj in value.items():
            #                     pj['abnormal'] = True
            #                     pj['alter'] = False
            #                     pj['starttime'] = c_time_int
            #             dic_state['checklate'] = True
            #
            #         end_c_json(dic, mqtt_tool, am_work, am_home, pm_work, pm_home,
            #                    am_work_add, pm_work_add, am_home_cut, pm_home_cut, dic_camera, ca_id, frame, current_time, offduty_limit, mask_item, read, q_frame_difference, fps)
            #
            #     duty_status = []
            # # view_video(frame, ca_id, start_time)
            # start_time = time.time()  # 初始化为当前时间
            else:  # if len(q) > 0:
                mqtt_tool.pubish(json.dumps(str('队列消费完了, 正在等待...')))
                print('队列消费完了, 正在等待...')
                time.sleep(10)
                continue


def r_move(fps, q_frame_all, frame_all_list, human_exists):
    # First Remove None person sits, have to use Reverse mode!
    human_exists_ori = human_exists
    for i in range(len(human_exists) - 1, -1, -1):
        iou = human_exists[i][1]['prob_iou']
        if iou < 1:  # remember to recover!!!!!!!!!!!
            # if (iou < 0):
            del human_exists[i]

    frame_sits = []
    frame_all = dict()
    while True:
        # print(len(q))
        if len(q_frame_all) > 0:
            try:
                vid_lis = q_frame_all.pop()
                # # continue
                # q_consume.insert(0, vid_lis)
            except:
                print("completed")
                return
                # continue # Remember to recover!
            num_sits = 0
            cn = vid_lis[2]
            # assert(cn >= 0 and cn < fps)

            if vid_lis != []:
                # frame_allone = vid_lis[1]
                num_sits = len(human_exists)
                frame_1group = []

                for h_e in human_exists:
                    position = h_e[1]['position']
                    # frame_1group.append(vid_lis[1][(int)(position[0]):(int)(position[2]), (int)(position[1]):(int)(position[3]), :])
                    frame_1group.append(
                        vid_lis[1][int(position[1]):int(position[3]), int(position[0]):int(position[2]), :])
                frame_sits.append(frame_1group)
                # frame_all.append(frame_allone)
                ca_id = vid_lis[0]
                if ca_id not in frame_all.keys():
                    frame_all[ca_id] = [vid_lis[1]]
                else:
                    frame_all[ca_id].append(vid_lis[1])

                # if(cn==fps-1):
                if (cn + 1) % fps == 0:
                    break
            else:  # if vid_lis != []
                time.sleep(0.01)
                continue
        else:  # if len(q_frame_all) > 0:
            time.sleep(1)
            # continue
            break

    if len(frame_sits) != fps:
        print("num of frame per slowfast is not fps! is %d" % len(frame_sits))
        return

    pred_move_reslist = m_r.run(frame_sits, human_exists)
    print("slowfast result is", pred_move_reslist)

    # 这里是逐帧保存数据，目前是存在一个字典里面  结构{camera_id：[frame, frame, frame, frame...]}
    for key in frame_all.keys():
        for c_frame in frame_all[key]:
            c_frame = utils.true_box_move(c_frame, human_exists_ori, human_exists, pred_move_reslist)
            # if key not in frame_all_dict.keys():
            #     frame_all_dict[key] = [c_frame]
            # else:
            #     frame_all_dict[key].append(c_frame)
            frame_all_list.append(c_frame)
        print("length of frame_all_dict[key] is %d" % (len(frame_all_list)))

    # 生成视频
    # if len(q_frame_all) <= fps*len(frame_all_dict.keys()):
    #     for key in frame_all_dict.keys():
    #         video_writer = cv2.VideoWriter('./res1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30,
    #                                        (frame_all_dict[key][0].shape[1], frame_all_dict[key][0].shape[0]))
    #         for item in frame_all_dict[key]:
    #             video_writer.write(item)
    #         video_writer.release()

    # if len(q_frame_all) <= fps*10:
    if cn >= 600:
        video_writer = cv2.VideoWriter('./res_tf_.avi', cv2.VideoWriter_fourcc(*'XVID'), 24,
                                       (frame_all_list[0].shape[1], frame_all_list[0].shape[0]))
        for item in frame_all_list:
            video_writer.write(item)
        video_writer.release()


# 向detectFrameList写入数字
def writeQueue(detectFrameList, top: int, fps, queue, frameDiff, frameWholeDataHuman):
    """
    向q中写入数据
    :param detectFrameList:
    :param top: 6000000
    :param fps: 帧率 64
    :param queue: 队列
    :param frameDiff: 帧差
    :param frameWholeDataHuman: 人头帧部分
    :return:
    函数完成之后可以实现,detectFrameList 保存的是视频id,间隔帧数据和帧差的一个list
    frameWithFrameDiff 头部插入摄像头id号,还有当前连续帧,帧编号
    """
    # 视频信息,video_info 是 north0001, MyVideo_2.mp4
    # read 信息是
    """
    video_info :[('north0001', './MyVideo_2.mp4')]
    read = {'site': 'sxxzspj',
    'camera_data': [{'cameraid': 'north0001', 'link': './MyVideo_2.mp4', 
    'duty_data': [{'position': '410,433,830,717',
    'dutyid': '001', 'offduty_limit': 0.5, 'am_gowork_time': '8:00:00', 'am_gohome_time': '12:00:00',
    'pm_gowork_time': '13:00:00', 'pm_gohome_time': '23:50:00', 'gowork_time_after': 10, 
    'gohome_time_before': 10, 'sleepduty_limit': 0.5, 'playmobile_limit': 15},
     {'position': '737,306,999,498', 'dutyid': '002', 'offduty_limit': 0.5, 'am_gowork_time': '8:00:00',
    'am_gohome_time': '12:00:00', 'pm_gowork_time': '13:00:00', 'pm_gohome_time': '23:50:00', 
    'gowork_time_after': 10, 'gohome_time_before': 10, 'sleepduty_limit': 0.5, 'playmobile_limit': 15}, 
    {'position': '798,181,960,342', 'dutyid': '003', 'offduty_limit': 0.5, 'am_gowork_time': '8:00:00', 
    'am_gohome_time': '12:00:00', 'pm_gowork_time': '13:00:00', 'pm_gohome_time': '23:50:00', 
    'gowork_time_after': 10, 'gohome_time_before': 10, 'sleepduty_limit': 0.5, 'playmobile_limit': 15}, 
    {'position': '992,217,1177,403', 'dutyid': '004', 'offduty_limit': 0.5, 'am_gowork_time': '8:00:00', 
    'am_gohome_time': '12:00:00', 'pm_gowork_time': '13:00:00', 'pm_gohome_time': '23:50:00', 
    'gowork_time_after': 10, 'gohome_time_before': 10, 'sleepduty_limit': 0.5, 'playmobile_limit': 15}]}]}
    """

    videoInfo, totalConfig = utils.read_config(cfg)
    count = 0
    if type(fps) != int:
        fps = int(fps)
    # 存储摄像头id和视频对象(cv2)
    vid_lis = []
    for info in videoInfo:
        cam_id, link = info
        vid = cv2.VideoCapture(link)  # 视频对象
        vid_lis.append((cam_id, vid))
    global s
    preConfigFile = totalConfig
    cnt_none = 0
    while True:
        current_f = open(cfg.config_json, 'r', encoding='utf-8')
        currentConfigFile = json.load(current_f)
        current_f.close()
        if currentConfigFile != preConfigFile:  # 检测配置文件是否更改, 如果更改,以当前的为准
            preConfigFile = currentConfigFile
            del currentConfigFile
            del detectFrameList[top:]  # top=>队列的长度限制，超出的删除
            gc.collect()  # 垃圾回收
            r_id = queue.get()
            os.kill(r_id, signal.SIGKILL)  # 杀死该进程
            run(fps)
        # 感觉和下面重复,其实没有重复,这里是为了读取每一帧的时候更新vid对象实现每隔fps帧检测一次;如果删掉会每循环25此检测连续帧

        for ca_id, vidObject in vid_lis:  # 读取每一个视频id和视频对象
            isVideo, frame = vidObject.read()  # 返回一个布尔值和视频的帧,布尔值True表示可以打开视频,如果读到结尾会返回false
            frameWithFrameDiff = list()  # 注意视频流处理v方式
            if isVideo:  # don't forget to remove
                # q_frame_all.append((ca_id, frame, cn))
                frameWholeDataHuman.insert(0, (ca_id, frame, count))  # 头部插入摄像头id号,还有当前连续帧,帧编号
                if count % fps == 0:
                    currentframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if len(frameDiff) > top:
                        del frameDiff[top:]

                    if len(frameDiff) > 0:
                        fl = False
                        for item_q in frameDiff:
                            for item_v in item_q:
                                if item_v[0] == ca_id:
                                    previousframe = cv2.cvtColor(item_v[1], cv2.COLOR_BGR2GRAY)
                                    fl = True
                                    break
                            if fl:
                                break
                    else:
                        previousframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    frameDiff_ = cv2.absdiff(currentframe, previousframe)

                    # cv2.imwrite('/home/yuhao/Desktop/model_read_json/video_surveillance/ai/res/' + str(time.time()) + '.jpg', mask)
                    frameWithFrameDiff.append((ca_id, frame, frameDiff_))
            else:
                print("%s 号 视频读取完毕" % ca_id)
                time.sleep(1)
                cnt_none += 1  # Still have problems, only fit for 1 video
                if cnt_none < 4:
                    continue
                else:
                    return
        count += 1
        # if(cn == fps): # Remember to recover!
        #     cn = 0
        if frameWithFrameDiff != []:

            detectFrameList.insert(0, frameWithFrameDiff)  # frameWithFrameDiff 保存的是视频id,间隔帧数据和帧差的一个list
            if len(detectFrameList) >= top:  # 清理内存垃圾
                del detectFrameList[top:]
                gc.collect()

        ## aa = False # only write 1 frame per camera, don't forget to remove!


# 1.在写函数里面插入v时，也将mask插入进去，缺陷:若读写速度不均，读快的话会导致得不到上一次视频取帧，也就得不到mask 废弃
# 2.再定义一个共享列表，这个列表存储读函数pop掉的数据
def run(fps):
    """不再检测时间范围时，进程如何处理"""
    mqtt_tool = MqttTool()  # mqtt目前没问题
    # mqtt_tool = None
    queue = Queue()
    # 存放检测帧的list数据
    detectFrameList = Manager().list()  # 进程之间共享list
    # 从detectFrameList  pop掉后会存入q_consume，目的是保存之前的数据可以做帧差
    frameDiff = Manager().list()

    # 所有帧数据，为了合成视频流
    frameWholeDataHuman = Manager().list()
    # frame_all_dict = Manager().dict()
    frameWholeDataMove = Manager().list()

    # human exist or not
    humanExistList = Manager().list()
    # 向detectFrameList中写入数据, 这一部分执行完成之后,
    # detectFrameList 保存的是视频id,间隔帧数据和帧差的一个list, frameWholeDataHuman 摄像头id号,连续帧,帧编号
    pw = Process(target=writeQueue, args=(detectFrameList, 6000000, fps, queue, frameDiff, frameWholeDataHuman))
    pw.start()

    while True:
        # 读取detectFrameList中的数据并进行human detecting
        # pr = Process(target=r, args=(mqtt_tool, q, queue, q_consume, fps, q_frame_all))
        pr_human = Process(target=r_human, args=(mqtt_tool, detectFrameList, queue, frameDiff, fps, frameWholeDataHuman, humanExistList))
        pr_move = Process(target=r_move, args=(fps, frameWholeDataHuman, frameWholeDataMove, humanExistList))

        pr_human.start()
        pr_human.join()
        pr_move.start()
        pr_move.join()

# 多路摄像头帧差法测试有问题
