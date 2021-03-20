import cv2
import time
import datetime
import json
import gc
import os
import numpy as np
from ai.core import utils
from ai import config as decfg
from tools.mqtt_tool import MqttTool
from multiprocessing import Process, Manager, Queue
import signal
from ai import move_recognition as m_r
import argparse
import colorsys
import torch
import torch.nn as nn
from ai.yolo4 import YoloBody
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFont, ImageDraw
from torch.autograd import Variable
import multiprocessing as mp
from utils.utils import non_max_suppression, bbox_iou, DecodeBox, letterbox_image, yolo_correct_boxes


class YOLO(object):
    _defaults = {
        # old version
        # "model_path1": "/home/bruce/bigVolumn/autolabelData/humanDetectionModel/checkpoints_2/Epoch20-Total_Loss2.3993-Val_Loss6.8344.pth",
        # "model_path2": "/home/bruce/bigVolumn/autolabelData/humanDetectionModel/checkpoints_4/Epoch20-Total_Loss2.6424-Val_Loss6.9749.pth",

        "model_path1": "/home/bruce/bigVolumn/autolabelData/humanDetectionModel/checkpoints_5/Epoch18-Total_Loss2.6925-Val_Loss6.3727.pth",
        "model_path2": "/home/bruce/bigVolumn/autolabelData/humanDetectionModel/checkpoints_5/Epoch19-Total_Loss2.5434-Val_Loss6.3093.pth",

        "anchors_path": '/home/bruce/bigVolumn/tempPycharmProjects/videoMonitor_v4/model_read_json/video_surveillance/ai/model_data/yolo_anchors_pt.txt',
        "classes_path": '/home/bruce/bigVolumn/tempPycharmProjects/videoMonitor_v4/model_read_json/video_surveillance/ai/model_data/voc_classes.txt',
        "model_image_size": (608, 608, 3),
        "confidence": 0.2,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):

        self.net1 = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()
        self.net2 = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()

        # 加快模型训练的效率
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict1 = torch.load(self.model_path1, map_location=device)
        state_dict2 = torch.load(self.model_path2, map_location=device)
        self.net1.load_state_dict(state_dict1)
        self.net2.load_state_dict(state_dict2)

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net1 = nn.DataParallel(self.net1)
            self.net1 = self.net1.cuda()
            self.net2 = nn.DataParallel(self.net2)
            self.net2 = self.net2.cuda()

        print('Finished!')

        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], len(self.class_names), (self.model_image_size[1], self.model_image_size[0])))

        # print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(0.09, 0.79, 0.591) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        # resize to 608 x 608 x 3
        crop_img = np.array(letterbox_image(image, (self.model_image_size[0], self.model_image_size[1])))
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs1 = self.net1(images)
            outputs2 = self.net2(images)
        output_list1 = []
        output_list2 = []
        for i in range(3):
            output_list1.append(self.yolo_decodes[i](outputs1[i]))
            output_list2.append(self.yolo_decodes[i](outputs2[i]))

        output1 = torch.cat(output_list1, 1)
        output2 = torch.cat(output_list2, 1)

        output = torch.cat((output1, output2), 1)
        batch_detections = non_max_suppression(output, len(self.class_names), conf_thres=self.confidence, nms_thres=0.3)
        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image

        top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
        top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
        top_label = np.array(batch_detections[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detections[top_index, :4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), \
                                                 np.expand_dims(top_bboxes[:, 1], -1), \
                                                 np.expand_dims(top_bboxes[:, 2], -1), \
                                                 np.expand_dims(top_bboxes[:, 3], -1)

        # 去掉灰条
        # 一般认为left, top是左上角的点坐标, right, bottom 是右下角的坐标
        # 注意这里输入的预测框的结果不是[xmin, ymin, xmax, ymax],而是[ymin, xmin, ymax, xmax]=[top, left, bottom, right]
        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                   np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)

        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
        return boxes, top_label, top_conf

# begin
start = time.time()
coco = decfg.coco
classes = utils.read_class_name(coco)  # 标签只有一类为{0: 'person'}

color_table = utils.get_color_table(1)

_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]


def video_obj_pytorch(yolo, frame, read, ca_id, dic_camera, q_frame_all, frame_all_dict, fps):
    box_dic = dict()
    box_dic['classes'] = classes
    box_dic['color_table'] = color_table

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(np.uint8(img))
    boxes_, classes_, scores_ = yolo.detect_image(img)

    box_dic['boxes_'] = boxes_
    box_dic['scores_'] = scores_
    box_dic['labels_'] = classes_
    # print('the box info is: \n', box_dic)

    predict_box_list = utils.predict_box(frame, box_dic, False)  # 过滤坐标
    # print('the predict box list is:', predict_box_list)
    # 获取实际坐标 true_box_list=》[(工位id, (min_x, min_y, max_x, max_y) )]    dutyid_list -> 工位id
    true_box_list, dutyid_list = utils.true_box(read, frame, ca_id, dic_camera)
    # print('true box size is: \n', true_box_list)

    res_list = utils.compute_status(true_box_list, predict_box_list, dutyid_list, ca_id)
    return res_list


def r_human(yolo, mqtt_tool, detectFrameList, queue, frameDiff, fps, frameWholeData, frameWholeMove, human_exists):
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
    q_frame_difference = dict()  # 存放帧差数组
    frame_all_dict = dict()  # 存放所有帧(为了合成视频流)

    video_info, read = utils.read_config(decfg)
    # print(video_info)
    # print(read)
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
    s_time = datetime.datetime.now().strftime('%H:%M:%S')
    # s_time = s_time.strftime('%H:%M:%S')

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
                camera_data = read['camera_data']
                for ca_id, frame, mask_item in vid_lis:  # 从上面的detectFrameList 表里面拿出一个进行测试
                    # flag等于True时处理脱岗离岗，等于False时处理迟到早退
                    flag = True  # None
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
                        time.sleep(2)
                        continue

                    # 预测人物框
                    res_list = video_obj_pytorch(yolo, frame, read, ca_id, dic_camera, frameWholeData, frame_all_dict,
                                                 fps)  # 计算的状态
                    duty_status.extend(res_list)
                    human_exists.extend(res_list)
                    r_move(fps, frameWholeData, frameWholeMove, human_exists)
                break  # while True:
            mqtt_tool.pubish(json.dumps(str('队列消费完了, 正在等待...')))
            print('队列消费完了, 正在等待...')
            time.sleep(2)
            continue


def r_move(fps, q_frame_all, frame_all_list, human_exists):
    """
    动作行为识别模块
    :param fps: 帧数
    :param q_frame_all: 所有视频帧
    :param frame_all_list: 保存动作帧数据
    :param human_exists: 检测到的人的坐标
    :return:
    """
    global start
    human_exists_ori = human_exists
    for i in range(len(human_exists) - 1, -1, -1):
        iou = human_exists[i][1]['prob_iou']
        if iou < 1:  # remember to recover!
            del human_exists[i]

    frame_sits = []
    frame_all = dict()
    while True:
        if len(q_frame_all) > 0:
            try:
                vid_lis = q_frame_all.pop()
            except:
                print("completed")
                return
                # continue # Remember to recover!
            num_sits = 0
            frameCount = vid_lis[2]
            if vid_lis != []:
                frame_1group = []
                for h_e in human_exists:
                    position = h_e[1]['position']
                    frame_1group.append(
                        vid_lis[1][int(position[1]):int(position[3]), int(position[0]):int(position[2]), :])
                frame_sits.append(frame_1group)
                # frame_all.append(frame_allone)
                ca_id = vid_lis[0]
                if ca_id not in frame_all.keys():
                    frame_all[ca_id] = [vid_lis[1]]
                else:
                    frame_all[ca_id].append(vid_lis[1])

                if (frameCount + 1) % fps == 0:
                    break
            else:  # if vid_lis != []
                time.sleep(0.01)
                continue
        else:  # if len(q_frame_all) > 0:
            # time.sleep(1)
            # continue
            break

    if len(frame_sits) != fps:
        return
    pred_move_reslist = m_r.run(frame_sits, human_exists)
    print("slowfast result is", pred_move_reslist, end="")
    print("\t", len(pred_move_reslist))

    # 这里是逐帧保存数据，目前是存在一个字典里面  结构{camera_id：[frame, frame, frame, frame...]}
    for key in frame_all.keys():
        for c_frame in frame_all[key]:
            c_frame = utils.true_box_move(c_frame, human_exists_ori, human_exists, pred_move_reslist)
            frame_all_list.append(c_frame)
        # print("length of frame_all_dict[key] is %d" % (len(frame_all_list)))
    # print('the cn is: ', cn)
    if frameCount >= 10:
        # print("begin to write the video clip into the local files!")
        video_writer = cv2.VideoWriter('/home/bruce/bigVolumn/autolabelData/demoOutputVideo/result/'
                                       '1101_5.avi', cv2.VideoWriter_fourcc(*'XVID'), 24,
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
    videoInfo, totalConfig = utils.read_config(decfg)
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
        current_f = open(decfg.config_json, 'r', encoding='utf-8')
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

                    frameWithFrameDiff.append((ca_id, frame, frameDiff_))
            else:
                print("%s 号 视频读取完毕" % ca_id)
                # time.sleep(1)
                cnt_none += 1  # Still have problems, only fit for 1 video
                if cnt_none < 4:
                    continue
                else:
                    return
        count += 1
        if frameWithFrameDiff.__len__() != 0:

            detectFrameList.insert(0, frameWithFrameDiff)  # frameWithFrameDiff 保存的是视频id,间隔帧数据和帧差的一个list
            if len(detectFrameList) >= top:  # 清理内存垃圾
                del detectFrameList[top:]
                gc.collect()


def run(fps):
    """
    1.在写函数里面插入v时，也将mask插入进去，缺陷:若读写速度不均，读快的话会导致得不到上一次视频取帧，也就得不到mask 废弃
    2.再定义一个共享列表，这个列表存储读函数pop掉的数据
    不再检测时间范围时，进程如何处理
    """
    # detectron2 config
    mp.set_start_method("spawn", force=True)
    # args = get_parser().parse_args()
    # cfg = setup_cfg(args)
    # predictor = DefaultPredictor(cfg)
    yolo = YOLO()

    mqtt_tool = MqttTool()  # mqtt目前没问题
    queue = Queue()
    # 存放检测帧的list数据
    detectFrameList = Manager().list()  # 进程之间共享list
    # 从detectFrameList  pop掉后会存入q_consume，目的是保存之前的数据可以做帧差
    frameDiff = Manager().list()

    # 所有帧数据，为了合成视频流
    frameWholeDataHuman = Manager().list()
    frameWholeDataMove = Manager().list()

    # human exist or not
    humanExistList = Manager().list()
    # 向detectFrameList中写入数据, 这一部分执行完成之后,
    # detectFrameList 保存的是视频id,间隔帧数据和帧差的一个list, frameWholeDataHuman 摄像头id号,连续帧,帧编号
    pw = Process(target=writeQueue, args=(detectFrameList, 6000000, fps, queue, frameDiff, frameWholeDataHuman))
    pw.start()

    while True:
        # 读取detectFrameList中的数据并进行human detecting
        pr_human = Process(target=r_human, args=(
        yolo, mqtt_tool, detectFrameList, queue, frameDiff, fps, frameWholeDataHuman, frameWholeDataMove,
        humanExistList))

        pr_human.start()
        pr_human.join()
