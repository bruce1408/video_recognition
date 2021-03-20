from __future__ import division, print_function
import cv2, time, datetime, json
import numpy as np
import tensorflow as tf
from ai.core import utils
from ai import config as cfg
from tools.mqtt_tool import MqttTool
import gc, os
from multiprocessing import Process, Manager, Queue
import signal
from tensorflow import saved_model as sm

s = 0
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
graph = tf.Graph()
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
path = cfg.path
anchors = cfg.anchors
coco = cfg.coco
anchors = utils.parse_anchors(anchors)
classes = utils.read_class_name(coco)
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
    '''计算工位状态'''
    # TODO
    if True:
        img, resize_ratio, dw, dh = utils.letterbox_resize(frame, input_size, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.
    box_dic = dict()
    boxes_, scores_, labels_ = run_sess(img, sess, SignatureDef)
    box_dic['boxes_'] = boxes_
    box_dic['resize_ratio'] = resize_ratio
    box_dic['dw'] = dw
    box_dic['dh'] = dh
    box_dic['scores_'] = scores_

    box_dic['labels_'] = labels_
    box_dic['classes'] = classes
    box_dic['color_table'] = color_table

    predict_box_list = utils.predict_box(frame, box_dic)  # 預測坐標(original frame)
    true_box_list, dutyid_list = utils.true_box(read, frame, ca_id, dic_camera)  # 获取实际坐标 true_box_list=》[(工位id, (min_x, min_y, max_x, max_y) )]    dutyid_list =>工位id
    res_list = utils.compute_status(true_box_list, predict_box_list, dutyid_list, ca_id)

    ####这里是逐帧保存数据，目前是存在一个字典里面    结构{camera_id：[frame, frame, frame, frame...]}
    del_list = []
    for c_data in range(len(q_frame_all)):
        if q_frame_all[c_data][0] == ca_id:
            del_list.append(c_data)
            c_frame = q_frame_all[c_data][1]
            # _ = utils.predict_box(c_frame, box_dic)
            for p in predict_box_list:
                utils.plot_one_box(c_frame, [p[1][0], p[1][1], p[1][2], p[1][3]],
                             label='person',
                             color=(255, 255, 255))


            _, _ = utils.true_box(read, c_frame, ca_id, dic_camera)
            if ca_id not in frame_all_dict.keys():
                frame_all_dict[ca_id] = [c_frame]
            else:
                frame_all_dict[ca_id].append(c_frame)
            if len(del_list) == fps:
                break

    del_list.sort(reverse=True)
    for i in del_list:
        del q_frame_all[i]

    #生成视频
    # print('fps---------->', type(fps))
    # print('len(frame_all_dict.keys())------>', type(len(frame_all_dict.keys())))
    # print('len(q_frame_all)------>',type(len(q_frame_all)))
    # print('fps*len(frame_all_dict.keys())------>',type(fps*len(frame_all_dict.keys())))
    if len(q_frame_all) <= fps*len(frame_all_dict.keys()):
        for key in frame_all_dict.keys():
            video_writer = cv2.VideoWriter('/home/yuhao/Desktop/model_read_json/video_surveillance/ai/res1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30,
                                           (frame_all_dict[key][0].shape[1], frame_all_dict[key][0].shape[0]))
            for item in frame_all_dict[key]:
                video_writer.write(item)
            video_writer.release()










    return res_list


def view_video(frame, ca_id, start_time):
    end_time = time.time()
    res_img = frame
    cv2.putText(res_img, 'cost: {:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,
                fontScale=1, color=(0, 0, 255), thickness=5)

    cv2.namedWindow(ca_id, cv2.WINDOW_NORMAL)
    cv2.imshow(ca_id, res_img)
    cv2.waitKey(1)



'''
脱岗离岗
dic:每个工位的状态(刚在视频帧上预测得到的，实时)
am_home_cut:上午早退阀值
pm_home_cut:下午早退阀值
dic_camera:每个工位的状态(自己定义的变量存储工位状态)
offduty_limit:检测不在岗状态多长时间为脱岗
c_time:当前时间(并非实时，而是计算各个指标时的时间)
frame:图像
'''
def end_t_json(dic, mqtt_tool, am_home_cut, pm_home_cut, dic_camera, ca_id, frame, c_time, offduty_limit, mask, config_dict, q_frame_difference, fps):
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


                flag_mask, sleepduty_limit, q_frame_difference = utils.compute_sleep(ki, kj, mask, config_dict, q_frame_difference)



                if len(q_frame_difference[ki + kj]) > int(sleepduty_limit * 60 * 30 / fps):
                    del q_frame_difference[ki + kj][int(sleepduty_limit * 60 * 30 / fps):]



                if len(q_frame_difference[ki + kj]) >= int(sleepduty_limit * 60 * 30 / fps) - 1:
                    sum_num = 0
                    for difference in q_frame_difference[ki + kj]:
                        if difference > 20: ###这里认为帧差超过20像素为有相对位移
                            sum_num += 1

                    if sum_num < int(sleepduty_limit * 60 * 30 / fps * 0.2): #####若sleepduty_limit分钟内检测的帧数中帧差超过20的数量小于检测总帧数的0.2则认为为睡觉状态
                        flag_mask = False

                if dic_camera[ki][kj]['sleep_start_time'] == None:
                    dic_camera[ki][kj]['sleep_start_time'] = st_time


                if flag_mask:
                    dic_camera[ki][kj]['sleep_start_time'] = st_time
                    dic_camera[ki][kj]['sleepduty'] = False
                    res_dic["sleep"] = "no"
                else:
                    sleep_time = utils.time_cha(dic_camera[ki][kj]['sleep_start_time'])
                    if sleep_time >= sleepduty_limit*60 and sleep_time <= sleepduty_limit*60 + 999999:  ###会在睡岗超过sleepduty_limit后警报999999s，然后初始化状态
                        dic_camera[ki][kj]['sleepduty'] = True
                        res_dic["sleep"] = "yes"
                    elif sleep_time > sleepduty_limit*60 + 999999:   ###警报超过999999s，初始化状态
                        dic_camera[ki][kj]['sleep_start_time'] = st_time
                        dic_camera[ki][kj]['sleepduty'] = False
                        res_dic["sleep"] = "no"

                if 'sleep' not in res_dic.keys():
                    res_dic["sleep"] = 'no'



            if not outduty:
                dic_camera[ki][kj]['starttime'] = st_time


            altertime = utils.time_cha(dic_camera[ki][kj]['starttime'])  ##altertime：脱岗离岗时间


            if c_time < am_home_cut - 10 or c_time < pm_home_cut - 10:  ###如果再工作时间内
                if altertime > float(offduty_limit)*60:  # TODO 实际生产中把 2 替换为=》float(offduty_limit)*60 - 20

                    if not dic_camera[ki][kj]['alter']:
                        dic_camera[ki][kj]['altertime'] = altertime
                        dic_camera[ki][kj]['alter'] = True
                        res_dic["status"] = "offduty"
                        dic_camera[ki][kj]['on_off_duty'] = "offduty"
                    # if utils.time_cha(dic['altertime']) <= 10: # 10s
                    if altertime > dic_camera[ki][kj]['altertime'] + 999999:  # 999999s     ####脱岗离岗时间超过阈值999999s，警报持续999999s
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
            cv2.putText(res_img, "camera_id:{0},timestemp:{1},cost:{2}".format(ki, time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                               time.localtime()),
                                                                             cost_time), (25, 25), font, 1, (0, 255, 0),
                        thickness=2, lineType=2)

            for index, resdata in enumerate(res_data):
                cv2.putText(res_img, json.dumps(resdata), (25, 25 * (index + 2)), font, 1, (0, 255, 0), thickness=2,
                            lineType=2)


            cv2.imwrite("/home/yuhao/Desktop/model_read_json/video_surveillance/ai/res/end_t_{0}.jpg".format(time.time() * 1000), res_img)

'''
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
'''
def end_c_json(dic, mqtt_tool, am_work, am_home, pm_work, pm_home,
               am_work_add, pm_work_add, am_home_cut, pm_home_cut, dic_camera, ca_id, frame, c_time, offduty_limit, mask, config_dict, q_frame_difference, fps):
    '''迟到早退'''
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
                sleepduty_limit, flag_mask, q_frame_difference = utils.compute_sleep(ki, kj, mask, config_dict, q_frame_difference)

                if len(q_frame_difference[ki][kj]) > int(sleepduty_limit * 60 * 30 / fps):
                    del q_frame_difference[ki][kj][int(sleepduty_limit * 60 * 30 / fps):]
                if len(q_frame_difference[ki][kj]) >= int(sleepduty_limit * 60 * 30 / fps):
                    sum_num = 0
                    for difference in q_frame_difference[ki][kj]:
                        if difference > 20:  ###这里认为帧差超过20像素为有相对位移
                            sum_num += 1
                    if sum_num < int(sleepduty_limit * 60 * 30 / fps * 0.2): ##若sleepduty_limit分钟内检测的帧数中帧差超过20的数量小于检测总帧数的0.2则认为为睡觉状态
                        flag_mask = False

                if dic_camera[ki][kj]['sleep_start_time'] == None:
                    dic_camera[ki][kj]['sleep_start_time'] = st_time

                if flag_mask:
                    dic_camera[ki][kj]['sleep_start_time'] = st_time
                    dic_camera[ki][kj]['sleepduty'] = False
                    res_dic["sleep"] = "no"
                else:
                    sleep_time = utils.time_cha(dic_camera[ki][kj]['sleep_start_time'])
                    if sleep_time >= sleepduty_limit*60 and sleep_time <= sleepduty_limit*60 + 999999:  ###会在睡岗超过sleepduty_limit后警报999999s，然后初始化状态
                        dic_camera[ki][kj]['sleepduty'] = True
                        res_dic["sleep"] = "yes"
                    elif sleep_time > sleepduty_limit*60 + 999999:   ###警报超过999999s，初始化状态
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
                    if altertime > float(offduty_limit)*60:  # TODO 实际生产中把 2 替换为=》float(offduty_limit)*60 - 20
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
                    if altertime > float(offduty_limit)*60:  # TODO 实际生产中把 2 替换为=》float(offduty_limit)*60 - 20
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
                                                                             cost_time), (25, 25), font, 1, (0, 255, 0),
                        thickness=2, lineType=2)

            for index, resdata in enumerate(res_data):
                cv2.putText(res_img, json.dumps(resdata), (25, 25 * (index + 2)), font, 1, (0, 255, 0), thickness=2,
                            lineType=2)

            # cv2.imwrite("/home/yuhao/Desktop/model_read_json/video_surveillance/ai/res/end_t_{0}.jpg".format(time.time() * 1000), res_img)
            mqtt_tool.pubish(json.dumps(dic_json))
            # print('this is json >>>: ', dic_json)
            # TODO end


def r(mqtt_tool, q, q2, q_consume, fps, q_frame_all):
    with tf.Session(graph=graph, config=config) as sess:
        MetaGraphDef = sm.loader.load(sess, tags=[sm.tag_constants.TRAINING], export_dir=path)
        # 解析得到 SignatureDef protobuf
        SignatureDef_d = MetaGraphDef.signature_def
        SignatureDef = SignatureDef_d[sm.signature_constants.CLASSIFY_INPUTS]

        q_frame_difference = dict()  ##存放帧差数组
        frame_all_dict = dict()      ##存放所有帧(为了合成视频流)

        video_info, read = utils.read_config(cfg)
        dic_state = dict()
        dic_state['checkout'] = False   #是否离岗
        dic_state['checklate'] = False  #是否迟到


        dic_camera = dict()
        dic_p = dict()
        dic_p['starttime'] = None
        dic_p['alter'] = False
        dic_p['altertime'] = None
        dic_p['abnormal'] = True
        dic_p['sleep_start_time'] = None
        dic_p['sleepduty'] = False
        dic_p['on_off_duty'] = 'onuty'


        for infos in read['camera_data']:
            camid = infos['cameraid']
            dic_camera[camid] = {
                duty['dutyid']: dic_p.copy() for duty in infos['duty_data']  ##每一个工位copy一个dic_p
            }
        start_time = time.time()
        s_time = datetime.datetime.now()
        s_time = s_time.strftime('%H:%M:%S')

        duty_status = []
        q2.put(os.getpid())  # 将消费进程r的id喂给队列，由生产进程w来获取
        while True:
            if len(q) > 0:
                try:
                    vid_lis = q.pop()
                    q_consume.insert(0, vid_lis)
                except:
                    continue
                if vid_lis != []:
                    camera_data = read['camera_data']
                    for ca_id, frame, mask_item in vid_lis:
                        flag = None  # (flag等于True时处理脱岗离岗，等于False时处理迟到早退)
                        duty_data = None
                        for cam in camera_data:
                            if cam['cameraid'] == ca_id:
                                duty_data = cam['duty_data']
                        for duty_data_dic in duty_data:
                            dutyid = duty_data_dic['dutyid']  # 工位id
                            offduty_limit = duty_data_dic['offduty_limit']  # 检测不在岗状态多长时间为脱岗
                            am_gowork_time = duty_data_dic['am_gowork_time']  # 上午上班时间
                            am_gohome_time = duty_data_dic['am_gohome_time']  # 上午下班时间
                            pm_gowork_time = duty_data_dic['pm_gowork_time']  # 下午上班时间
                            pm_gohome_time = duty_data_dic['pm_gohome_time']  # 下午下班时间
                            gowork_time_after = duty_data_dic['gowork_time_after']  # 上班后检测多久
                            gohome_time_before = duty_data_dic['gohome_time_before'] # 下班前多久检测
                            sleepduty_limit = duty_data_dic['sleepduty_limit']  # 超过多久算睡觉
                            playmobile_limit = duty_data_dic['playmobile_limit']  # 超过多久算玩手机

                            current_time = int(time.time())  # 当前时间
                            time_date = time.strftime("%Y%m%d")  # 当前日期
                            # 上午上班时间
                            am_work = int(time.mktime(
                                time.strptime(time_date + ''.join(str(am_gowork_time).split(':')), "%Y%m%d%H%M%S")))
                            # 上午迟到阀值
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
                        if flag != True and flag != False:
                            dic_state['checkout'] = False
                            dic_state['checklate'] = False
                            print('不在检测时间范围内')
                            time.sleep(10)
                            continue



                        res_list = video_obj(frame, sess, read, ca_id, SignatureDef, dic_camera, q_frame_all, frame_all_dict, fps)  # 计算的状态
                        duty_status.extend(res_list)
                        l_time = utils.time_cha(s_time)  # 计算时间差
##已经得到每个工位的状态
                        if int(l_time) >= 2:  ###### 2秒汇总一次(可以调节，这里指的是每隔几秒统计一次结果)
                            s_time = datetime.datetime.now()
                            s_time = s_time.strftime('%H:%M:%S')  # 重新初始化为当前时间
                            # dic = utils.compute_duty(duty_status)  ####  duty_status这里是 两秒内所有帧/fps 帧数和，若这些帧中某一个工位只要检测到一次人就证明此人在岗
                            if flag is True:   ##脱岗离岗
                                if not dic_state['checkout']:
                                    t_time_int = datetime.datetime.now()
                                    t_time_int = t_time_int.strftime('%H:%M:%S')
                                    for key, value in dic_camera.items():
                                        for pi, pj in value.items():
                                            pj['abnormal'] = True
                                            pj['alter'] = False
                                            pj['starttime'] = t_time_int

                                    dic_state['checkout'] = True
                                    dic_state['checklate'] = False
                                end_t_json(dic, mqtt_tool, am_home_cut, pm_home_cut, dic_camera, ca_id, frame, current_time, offduty_limit, mask_item, read, q_frame_difference, fps)
                            elif flag is False:   ##迟到早退
                                if not dic_state['checklate']:
                                    c_time_int = datetime.datetime.now()
                                    c_time_int = c_time_int.strftime('%H:%M:%S')
                                    for key, value in dic_camera.items():
                                        for pi, pj in value.items():
                                            pj['abnormal'] = True
                                            pj['alter'] = False
                                            pj['starttime'] = c_time_int
                                    dic_state['checklate'] = True

                                end_c_json(dic, mqtt_tool, am_work, am_home, pm_work, pm_home,
                                           am_work_add, pm_work_add, am_home_cut, pm_home_cut, dic_camera, ca_id, frame, current_time, offduty_limit, mask_item, read, q_frame_difference, fps)

                            duty_status = []
                        # view_video(frame, ca_id, start_time)
                        start_time = time.time()  # 初始化为当前时间
            else:
                mqtt_tool.pubish(json.dumps(str('队列消费完了, 正在等待...')))
                print('队列消费完了, 正在等待...')
                time.sleep(60)
                continue

'''
向q中写入数据
'''
def w(q, top: int, fps, q2, q_consume, q_frame_all):
    video_info, read = utils.read_config(cfg)
    cn = 0
    if type(fps) != int:
        fps = int(fps)
    ##存储摄像头id和视频对象(cv2)
    vid_lis = []
    for info in video_info:
        cam_id, link = info
        vid = cv2.VideoCapture(link)
        vid_lis.append((cam_id, vid))
    global s
    pred_read = read
    # tt = True
    while True:
        current_f = open(cfg.config_json, 'r', encoding='utf-8')
        current_read = json.load(current_f)
        current_f.close()
        if current_read != pred_read:  # 检测配置文件是否更改
            pred_read = current_read
            del current_read
            del q[top:] ########## top=>队列的长度限制，超出的删除
            gc.collect()
            r_id = q2.get()
            os.kill(r_id, signal.SIGKILL)
            run(fps)


        for ca_id, vid in vid_lis: ###########感觉和下面重复了 没有重复  这里是为了读取每一帧的时候更新vid对象实现每隔fps帧检测一次;如果删掉会每循环25此检测连续帧
            ret, frame = vid.read()
            if ret:
                q_frame_all.append((ca_id, frame))



        if cn % fps == 0:
            v = list()  # 注意视频流处理方式

            ##每个视频流取一帧
            for ca_id, vid in vid_lis:
                ret, frame = vid.read()
                if ret:
                    currentframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if len(q_consume) > top:
                        del q_consume[top:]

                    if len(q_consume) > 0:
                        fl = False
                        for item_q in q_consume:
                            for item_v in item_q:
                                if item_v[0] == ca_id:
                                    previousframe = cv2.cvtColor(item_v[1], cv2.COLOR_BGR2GRAY)
                                    fl = True
                                    break
                            if fl:
                                break
                    else:
                        previousframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    mask = cv2.absdiff(currentframe, previousframe)


                    # cv2.imwrite('/home/yuhao/Desktop/model_read_json/video_surveillance/ai/res/' + str(time.time()) + '.jpg', mask)
                    v.append((ca_id, frame, mask))
                else:
                    # print('没有帧了')
                    continue

            if v != []:
                q.insert(0, v)
                if len(q) >= top:  # 清理内存垃圾
                    del q[top:]
                    gc.collect()

            # tt = False # only write 1 groups, don't forget remove later!
        cn += 1


###1.在写函数里面插入v时，也将mask插入进去，缺陷:若读写速度不均，读快的话会导致得不到上一次视频取帧，也就得不到mask   废弃
###2.再定义一个共享列表，这个列表存储读函数pop掉的数据



def run(fps):
    '''不再检测时间范围时，进程如何处理'''
    mqtt_tool = MqttTool()  ############mqtt目前没问题
    # mqtt_tool = None
    q2 = Queue()
    ##存放检测帧的数据
    q = Manager().list()
    ##从q  pop掉后会存入q_consume，目的是保存之前的数据可以做帧差
    q_consume = Manager().list()

    ##所有帧数据，为了合成视频流
    q_frame_all = Manager().list()

    ##向q中写入数据
    pw = Process(target=w, args=(q, 300, fps, q2, q_consume, q_frame_all))
    ##读取q中的数据并进行识别
    pr = Process(target=r, args=(mqtt_tool, q, q2, q_consume, fps, q_frame_all))
    pw.start()
    pr.start()

    pw.join()
    pr.join()
    pw.terminate()
    pr.terminate()
