import cv2, json
import base64
import random
import numpy as np
import datetime


def time_cha(s_time):
    """
    计算时间差，参数时间值与当前时间
    """
    e_time = datetime.datetime.now()
    e_time = e_time.strftime('%H:%M:%S')
    d1 = datetime.datetime.strptime(s_time, '%H:%M:%S')
    d2 = datetime.datetime.strptime(e_time, '%H:%M:%S')
    try:
        # 时间差
        delta = str(d2 - d1)  # d2必须大于d1
        delta = datetime.datetime.strptime(delta, '%H:%M:%S')
        h = delta.hour
        m = delta.minute
        s = delta.second
        l_time = int(h) * 3600 + int(m) * 60 + int(s)
    except:
        raise ValueError("79行, now d1 > d2, but must to be d1 < d2")
    return l_time


def compute_duty(duty_status):
    """
    是否在岗综合分析
    duty_status:每个工位的状态
    是否可以融合在compute_status之中省略此方法
    不可以融合，本方法是对一些帧的预测结果的累加，所以说 offduty/onduty 属于[0,帧数]之间的整数
    """
    camera_dic = dict()
    dic = dict()
    '''
    dic变量的数据结构
    {
        'camera_id':{
            'dutyid':{
                'onduty' : 0,
                'offduty' : 0,
                'prob_iou' : 0
            }
        }
    }
    '''
    for c_id, value in duty_status:
        if camera_dic.get(c_id, -1) == -1:   # 若字典camera_dic中不存在key为c_id的键值对   c_id=>摄像头id
            camera_dic[c_id] = []
        camera_dic[c_id].append(value)
    for k in camera_dic:
        if dic.get(k, -1) == -1:
            dic[k] = dict()
        for duty_info in camera_dic[k]:
            if dic[k].get(duty_info['dutyid'], -1) == -1:
                dic[k][duty_info['dutyid']] = dict()
                dic[k][duty_info['dutyid']]['onduty'] = 0
                dic[k][duty_info['dutyid']]['offduty'] = 0
                dic[k][duty_info['dutyid']]["prob_iou"] = 0
            if dic[k].get(duty_info['dutyid'], -1) != -1 and duty_info['prob_iou'] == 1:
                dic[k][duty_info['dutyid']]['onduty'] += 1
                dic[k][duty_info['dutyid']]["prob_iou"] = duty_info['prob_iou']
                dic[k][duty_info['dutyid']]['offduty'] = 0
            elif dic[k].get(duty_info['dutyid'], -1) and duty_info['prob_iou'] == 0:
                dic[k][duty_info['dutyid']]['offduty'] += 1
                dic[k][duty_info['dutyid']]['onduty'] = 0
    print('the core utils part comput status is: ', dic)
    return dic


def predict_box(frame, box_dic: dict, imgProcess=False) -> list:
    """
    坐标预测
    :param frame: 视频提取的帧数据
    :param box_dic: 表示的是图片预测的结果和图片信息,保存在字典里面;
    :return:
    """
    classes = box_dic['classes']
    color_table = box_dic['color_table']
    # resize_ratio = box_dic['resize_ratio']
    # dw = box_dic['dw']
    # dh = box_dic['dh']
    boxes_ = box_dic['boxes_']
    scores_ = box_dic['scores_']
    labels_ = box_dic['labels_']
    predict_box_list = list()
    # if imgProcess:
    #     boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
    #     boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        class_id = labels_[i]
        if class_id == 0:  # 之前是if class_id == 1: 表示检测到了人
            plot_one_box(frame, [x0, y0, x1, y1],
                         label=classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                         color=color_table[labels_[i]])
            # draw_bbox(frame, bbox, cfg.coco_names)

            predict_box = (x0, y0, x1, y1)
            predict_box_list.append((class_id, predict_box, frame))
        else:
            predict_box = (1, 1, 1, 1)
            predict_box_list.append((class_id, predict_box, frame))
    if len(predict_box_list) == 0:
        predict_box_list.append((1, (0, 0, 0, 0), frame))
    return predict_box_list


def true_box(read, frame, ca_id, dic_camera):
    """
    获取实际坐标
    :param read:
    :param frame:
    :param ca_id:
    :param dic_camera:
    :return:
    """

    camera_data = read['camera_data']
    duty_data = None
    for cam in camera_data:
        if cam['cameraid'] == ca_id:
            duty_data = cam['duty_data']
    true_box_list = []
    dutyid_list = []
    for duty_data_lis in duty_data:
        position = duty_data_lis['position'].strip().split(',')
        min_x, min_y, max_x, max_y = float(position[0].strip()), float(position[1].strip()), float(
            position[2].strip()), float(position[3].strip()),
        dutyid = duty_data_lis['dutyid']
        dutyid_list.append(dutyid)

        # 在检测帧的图像上画框
        draw_bounding_box1(frame, round(min_x), round(min_y), round(max_x), round(max_y), dic_camera[ca_id][dutyid])
        # 真实的绝对坐标值
        true_box = (min_x, min_y, max_x, max_y)
        true_box_list.append((dutyid, true_box))
    return true_box_list, dutyid_list


def true_box_move(c_frame, human_exists_ori, human_exists, move_result):
    """
    according human detect move to classify results, Drawing boxes and texts to frame
    :param c_frame:
    :param human_exists_ori:
    :param human_exists:
    :param move_result:
    :return:
    """
    assert(len(human_exists) == len(move_result))
    if len(human_exists) != len(move_result):
        print("num of human and move is not the same!")
        return

    # draw non_person sits first ,remember to recover !
    for i in range(len(human_exists_ori) - 1, -1, -1):
        iou = human_exists_ori[i][1]['prob_iou']
        if iou < 1:
            color = (255, 0, 0)
            label = "status: no_person"
            position = human_exists_ori[i][1]['position']
            min_x, min_y, max_x, max_y = position[0], position[1], position[2], position[3]
            cv2.rectangle(c_frame, ((int)(min_x), (int)(max_y)), ((int)(max_x), (int)(min_y)), color, 2)
            cv2.putText(c_frame, label, ((int)(min_x + 3), (int)(min_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            # cv2.putText(c_frame, label, (min_x + 3, max_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # draw person_sits second
    for i in range(len(human_exists)):
        color = (0, 255, 0)
        label = move_result[i]
        position = human_exists[i][1]['position']
        min_x, min_y, max_x, max_y = position[0], position[1], position[2], position[3]
        cv2.rectangle(c_frame, ((int)(min_x), (int)(min_y)), ((int)(max_x), (int)(max_y)), color, 2)
        cv2.putText(c_frame, label, ((int)(min_x + 3), (int)(min_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 64), 2)
        # cv2.putText(c_frame, label, ((int)(min_x + 3), (int)(max_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return c_frame


def compute_status(true_box_list, predict_box_list, dutyid_list, ca_id):
    """
    根据工位坐标和预测标签计算iou，来得到每个工位的状态

    true_box_list:      真实坐标（工位坐标）      true_box_list -> [(工位id, (min_x, min_y, max_x, max_y) )]
    predict_box_list:   预测坐标                (class_id, predict_box, frame)
    dutyid_list:        工位id列表
    ca_id:              摄像头id
    """
    # print('the true box list is:', true_box_list)
    res_list = []
    for t_box in true_box_list:
        count = 0
        for p_box in predict_box_list:
            iou = broad_iou(t_box, p_box)
            count += iou
        if count >= 1:  # 不管几个人超过阈值iou都算作1
            res_list.append((ca_id, {'dutyid': t_box[0], 'position': t_box[1], 'status': 'on_duty', 'prob_iou': 1}))
        else:
            res_list.append((ca_id, {'dutyid': t_box[0], 'position': t_box[1], 'status': 'off_duty', 'prob_iou': 0}))
    return res_list


def compute_sleep(camera_id, duty_id, mask, config_dict, q_frame_difference, max_num=20):
    """
    根据帧差法的mask来判断工位上的人是否运动
    max_num:    工位位置上相对于上一帧之间不同的像素点个数限制，若超过此值则说明此工位位置内有运动
    :param camera_id:
    :param duty_id:
    :param mask:
    :param config_dict:
    :param q_frame_difference:
    :param max_num:
    :return: true==》有运动  false==>无运动
    """
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0
    sleepduty_limit = 0
    camera_data = config_dict['camera_data']
    for camera in camera_data:
        if camera_id == camera['cameraid']:
            for duty in camera['duty_data']:
                if duty_id == duty['dutyid']:
                    x_min = int(duty['position'].strip().split(',')[0].strip())
                    y_min = int(duty['position'].strip().split(',')[1].strip())
                    x_max = int(duty['position'].strip().split(',')[2].strip())
                    y_max = int(duty['position'].strip().split(',')[3].strip())
                    sleepduty_limit = float(duty['sleepduty_limit'])

    if x_min == 0 and y_min == 0 and x_max == 0 and y_max == 0:   # 若在配置文件中找不到该工位信息则返回没有睡觉
        return True, sleepduty_limit, q_frame_difference
    sum = 0
    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            if mask[i][j] > 50:
                sum += 1
    if camera_id + duty_id in q_frame_difference.keys():
        q_frame_difference[camera_id + duty_id].insert(0, sum)
    else:
        q_frame_difference[camera_id + duty_id] = [sum]

    if sum < max_num:
        return False, sleepduty_limit, q_frame_difference
    else:
        return True, sleepduty_limit, q_frame_difference


def read_config(cfg):
    """
    读取配置文件
    link 表示摄像头视频文件所存的位置
    cameraid 表示视频的id号
    video_info:[[摄像头id，视频流地址(或者本地视频地址)]]
    read:整个配置文件对象,是以字典的形式表示
    :param cfg:
    :return:
    """
    f = open(cfg.config_json, 'r', encoding='utf-8')
    read = json.load(f)
    f.close()
    camera_num = read['camera_data']
    video_info = []
    for k, camera_dic in enumerate(camera_num):
        link = camera_dic['link']
        cameraid = camera_dic['cameraid']
        video_info.append((cameraid, link))
    return video_info, read


def read_config_json():
    import time
    frame_all_dict = dict()  # 存放所有帧(为了合成视频流)
    path = "/home/bruce/PycharmProjects/videoMonitor/5.5/model_read_NoLive_human_move/model_read_json/video_surveillance/ai/config.json"
    f = open(path, 'r', encoding='utf-8')
    read = json.load(f)
    f.close()
    camera_num = read['camera_data']
    video_info = []
    for k, camera_dic in enumerate(camera_num):
        link = camera_dic['link']
        cameraid = camera_dic['cameraid']
        video_info.append((cameraid, link))

    print('the video_info: ', video_info)  # 视频信息
    print('the read is: ', read)  # json配置文件完整信息
    dic_state = dict()
    dic_state['checkout'] = False  # 是否离岗
    dic_state['checklate'] = False  # 是否迟到

    dic_camera = dict()
    dic_p = dict()
    dic_p['starttime'] = None
    dic_p['alter'] = False
    dic_p['altertime'] = None
    dic_p['abnormal'] = True
    dic_p['sleep_start_time'] = None
    dic_p['sleepduty'] = False
    dic_p['on_off_duty'] = 'onuty'

    # 摄像头工作坐标信息初始化
    for infos in read['camera_data']:
        camid = infos['cameraid']
        # 每一个工位copy一个dic_p
        dic_camera[camid] = {duty['dutyid']: dic_p.copy() for duty in infos['duty_data']}
        # print('the dic_camera is: ', dic_camera[camid])
        # print('the duty_data is: ', infos['duty_data'])
    print('the dic_camera is: ', dic_camera[camid])
    print('the duty_data is: ', infos['duty_data'])
    print('dic p is: ', dic_p)
    start_time = time.time()
    s_time = datetime.datetime.now()
    s_time = s_time.strftime('%H:%M:%S')


# read_config_json()


def cv2_base64(st):
    """
    转换图片为base64编码字符串,然后注意frame已经转换为BGR
    :param st:
    :return:
    """

    frame = cv2.cvtColor(st, cv2.COLOR_RGB2BGR)
    base64_str = cv2.imencode(".jpg", frame)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str


def broad_iou(t_box, p_box, min_iou=0.3):
    """
    计算iou
    min_iou：在岗iou阈值, the common of preict and gongwei // (gongwei + 0.1)
    :param t_box:
    :param p_box:
    :param min_iou:
    :return:
    """
    t_label, tbox = t_box
    p_label, pbox, f = p_box
    t_min_x, t_min_y, t_max_x, t_max_y = tbox
    p_min_x, p_min_y, p_max_x, p_max_y = pbox
    if p_min_x < t_max_x and p_max_x > t_min_x and p_min_y < t_max_y and p_max_y > t_min_y:
        x1 = max(t_min_x, p_min_x)
        y1 = max(t_min_y, p_min_y)
        x2 = min(t_max_x, p_max_x)
        y2 = min(t_max_y, p_max_y)
        intersect = (x2 - x1) * (y2 - y1)
        iou = intersect / ((p_max_y - p_min_y) * (p_max_x - p_min_x) + 0.1)
        if iou < min_iou:
            iou = 0
        else:
            iou = 1
    else:
        intersect = 0
        iou = intersect
    return iou


def draw_bounding_box1(img, x, y, x_plus_w, y_plus_h, duty):
    """
    画框
    :param img:
    :param x:
    :param y:
    :param x_plus_w:
    :param y_plus_h:
    :param duty:
    :return:
    """
    if duty['sleepduty']:
        label = duty['on_off_duty'] + ',yes_sleep'
    else:
        label = duty['on_off_duty'] + ',no_sleep'
    bbox_thick = int(0.6 * (y_plus_h + x_plus_w + 2000) / 50)
    t_size = cv2.getTextSize(label, 0, 1, thickness=bbox_thick // 2)[0]
    # cv2.rectangle(img, (x, y), (x + t_size[0] + 100, y - t_size[1] - 3), (0, 255, 0), -1)  # filled
    color = (0, 255, 0)
    if duty['on_off_duty'] != 'onduty':
        color = (255, 0, 0)
    else:
        if duty['sleepduty']:
            color = (0, 0, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    (w, h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.rectangle(img, (x, y - (2 * baseline + 5)), (x+w, y), (0, 255, 255), -1)  # 字体背景色
    cv2.putText(img, label, (x + 3, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 191), 1)  # 蓝色字体


def letterbox_resize(img, new_width, new_height, interp=0):
    """
    resize保持纵横比的情况下的把原来的image图像缩放到新的宽长尺寸的图片,不够的地方使用128来填充.
    :param img: 输入opencv图像矩阵
    :param new_width:缩放的宽度
    :param new_height:缩放的高度
    :param interp:差值方式,默认选择的是0,
    :return:返回值新的new_w,new_h的图像矩阵,缩放比例,还有上下,左右两侧填充的值
    """

    ori_height, ori_width = img.shape[:2]
    resize_ratio = min(new_width / ori_width, new_height / ori_height)
    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)
    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)  # 缩放到宽长比相同的尺寸下
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)  # 用128数字来填充这个图像.
    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)
    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img
    return image_padded, resize_ratio, dw, dh


def parse_anchors(anchor_path):
    """
    获取anchors
    parse anchors.
    returned data: shape [N, 2], dtype float32
    """
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors


def read_class_name(class_name_path):
    """
    获取names类别列表
    :param class_name_path:
    :return:
    """
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_color_table(class_num, seed=2):
    """
    获取颜色 随机
    输入的是class类别数目,然后随机生成3个数字范围在[0-255]之间,这三个随机数字保存和类别数一起
    返回一个字典{0:[x1, x2, x3]}
    :param class_num:
    :param seed:
    :return: {0:[x1, x2, x3]}
    """
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table


def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    """
    画框
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    :param img:
    :param coord:
    :param label:
    :param color:
    :param line_thickness:
    :return:
    """

    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 255, 191], thickness=tf, lineType=cv2.LINE_AA)
