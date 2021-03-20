import cv2, json
import base64
import random
import numpy as np
import datetime

###不包含帧差法


'''
计算时间差，参数时间值与当前时间
'''
def time_cha(s_time):

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

'''
是否在岗综合分析
duty_status:每个工位的状态

是否可以融合在compute_status之中省略此方法
'''
def compute_duty(duty_status):
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
        if camera_dic.get(c_id, -1) == -1:   ###若字典camera_dic中不存在key为c_id的键值对   c_id=>摄像头id
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
            elif dic[k].get(duty_info['dutyid'], -1) != -1 and duty_info['prob_iou'] >= 0.5:
                dic[k][duty_info['dutyid']]['onduty'] += 1
                dic[k][duty_info['dutyid']]["prob_iou"] = duty_info['prob_iou']
                dic[k][duty_info['dutyid']]['offduty'] = 0
            elif dic[k].get(duty_info['dutyid'], -1) and duty_info['prob_iou'] < 0.5:
                dic[k][duty_info['dutyid']]['offduty'] += 1
                dic[k][duty_info['dutyid']]['onduty'] = 0
    return dic

'''
用于darknet转化的tensorflow版本的应用
'''
def compute(sess, image_data, return_tensors, num_classes):
    image_data = image_data[np.newaxis, ...]
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
        feed_dict={return_tensors[0]: image_data})

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    return pred_bbox

'''
坐标的预测
'''
def predict_box(frame, box_dic):
    boxes_ = box_dic['boxes_']
    resize_ratio = box_dic['resize_ratio']
    dw = box_dic['dw']
    dh = box_dic['dh']
    scores_ = box_dic['scores_']
    labels_ = box_dic['labels_']
    classes = box_dic['classes']
    color_table = box_dic['color_table']
    # 預測坐標
    predict_box_list = []
    if True:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio

    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        class_id = labels_[i]
        if class_id == 1:
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

'''
获取实际坐标
'''
def true_box(read, frame, ca_id):

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
        draw_bounding_box1(frame, round(min_x), round(min_y), round(max_x), round(max_y))
        # 真实的绝对坐标值
        true_box = (min_x, min_y, max_x, max_y)
        true_box_list.append((dutyid, true_box))
    return true_box_list, dutyid_list

'''
根据工位坐标和预测标签计算iou，来得到每个工位的状态   #######逻辑混乱!！!遍历完工位以后就没必要进行下面操作，直接拼接结果即可
'''
def compute_status(true_box_list, predict_box_list, dutyid_list, ca_id):
    tp = list()
    for t_box in true_box_list:
        for p_box in predict_box_list:
            iou = broad_iou(t_box, p_box)
            tp.append((t_box[0], iou))
    on_duty_set = []
    off_duty_set = []
    for n in dutyid_list:   ####可能存在冗余
        for m in tp:
            if m[0] == n and m[1] > 0:
                on_duty_set.append(n)
            elif m[0] == n and m[1] == 0:
                off_duty_set.append(n)

    data = []
    for tup in tp:
        if tup[0] in on_duty_set:
            data.append({'dutyid': tup[0], 'status': 'on_duty', 'prob_iou': tup[1]})
        elif tup[0] in off_duty_set:
            data.append({'dutyid': tup[0], 'status': 'off_duty', 'prob_iou': tup[1]})


    ##每个工位最大iou(有时一个工位两个人)
    max_dict = {}
    for i in range(len(data)):
        if data[i]['dutyid'] not in max_dict.keys():
            max_dict[data[i]['dutyid']] = [i, data[i]['prob_iou']]
        else:
            if data[i]['prob_iou'] > max_dict[data[i]['dutyid']][1]:
                max_dict[data[i]['dutyid']] = [i, data[i]['prob_iou']]


    #将最大iou集合整理为最终结果           这一步有必要吗？是为了去重吗？为了去重的话完全没有必要在这进行两次循环？
    res_list = []
    for key in max_dict.keys():
        item = data[max_dict[key][0]]
        if item['prob_iou'] > 0:  # 设置在岗iou阈值
            item['status'] = 'on_duty'
        else:
            item['status'] = 'off_duty'
        res_list.append((ca_id, item))
    return res_list

'''
读取配置文件
video_info:[[摄像头id，视频流地址(或者本地视频地址)]]
read:整个配置文件对象(type==dict)
'''

def read_config(cfg):
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

'''
转换图片为base64编码字符串, 注意frame已转换为BGR格式
'''
def cv2_base64(st):

    frame = cv2.cvtColor(st, cv2.COLOR_RGB2BGR)
    base64_str = cv2.imencode(".jpg", frame)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str

'''
计算iou
'''
def broad_iou(t_box, p_box):
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
        if iou < 0.3:
            iou = 0
    else:
        intersect = 0
        iou = intersect
    return iou

'''
画框
'''
def draw_bounding_box1(img, x, y, x_plus_w, y_plus_h):
    label = ''
    bbox_thick = int(0.6 * (y_plus_h + x_plus_w + 2000) / 50)
    t_size = cv2.getTextSize(label, 0, 1, thickness=bbox_thick // 2)[0]
    cv2.rectangle(img, (x, y), (x + t_size[0] + 100, y - t_size[1] - 3), (0, 255, 0), -1)  # filled

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)
    cv2.putText(img, label, (x + 3, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

'''
resize保持纵横比的情况下resize
'''
def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]

    resize_ratio = min(new_width / ori_width, new_height / ori_height)

    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

    return image_padded, resize_ratio, dw, dh

'''
获取anchors
'''
def parse_anchors(anchor_path):
    '''
    parse anchors.
    returned data: shape [N, 2], dtype float32
    '''
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors

'''
获取names(类别列表)
'''
def read_class_name(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

'''
获取颜色(随机)
'''
def get_color_table(class_num, seed=2):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table

'''
画一个框
'''
def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
