from time import time
import numpy as np
import pandas as pd
import cv2
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
from slowfast.utils import logging
from slowfast.utils import misc
from slowfast.datasets import cv2_transform
from slowfast.models import model_builder
from slowfast.datasets.cv2_transform import scale

logger = logging.get_logger(__name__)
np.random.seed(20)


class VideoReader(object):

    def __init__(self, cfg):
        self.source = cfg.DEMO.DATA_SOURCE
        self.display_width = cfg.DEMO.DISPLAY_WIDTH
        self.display_height = cfg.DEMO.DISPLAY_HEIGHT
        try:  # OpenCV needs int to read from webcam
            self.source = int(self.source)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.source)
        if self.display_width > 0 and self.display_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        else:
            self.display_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.display_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.source))
        return self

    def __next__(self):
        was_read, frame = self.cap.read()
        if not was_read:
            # raise StopIteration
            ## reiterate the video instead of quiting.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None

        return was_read, frame

    def clean(self):
        self.cap.release()
        cv2.destroyAllWindows()


def demo(cfg, frame_sits, human_exists):
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    weightsSum = True
    # Setup logging format.
    logging.setup_logging()

    # Print config.
    # logger.info("Run demo with config:")
    # logger.info(cfg)
    # Build the video model and print model statistics.
    model1 = model_builder.build_model(cfg)
    model1.eval()
    model2 = model_builder.build_model(cfg)
    model2.eval()
    model3 = model_builder.build_model(cfg)
    model3.eval()
    # misc.log_model_info(model)
    # Load a checkpoint to test if applicable.
    ckptFlag = False
    if cfg.TEST.CHECKPOINT_FILE_PATH_1 != "" and cfg.TEST.CHECKPOINT_FILE_PATH_2 != "":
        ckpt1 = cfg.TEST.CHECKPOINT_FILE_PATH_1
        ckpt2 = cfg.TEST.CHECKPOINT_FILE_PATH_2
        ckpt3 = cfg.TEST.CHECKPOINT_FILE_PATH_3
        ckpt = [ckpt1, ckpt2, ckpt3]
        ckptFlag = True
    elif cfg.TEST.CHECKPOINT_FILE_PATH_1 != "":
        ckpt = cfg.TEST.CHECKPOINT_FILE_PATH_1
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        ckpt = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH_1 != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        ckpt = cfg.TRAIN.CHECKPOINT_FILE_PATH
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")
    cu.load_checkpoint(
        ckpt,
        model1,
        model2,
        model3,
        cfg.NUM_GPUS > 1,
        None,
        inflation=False,
        convert_from_caffe2="caffe2" in [cfg.TEST.CHECKPOINT_TYPE, cfg.TRAIN.CHECKPOINT_TYPE],
        ckptFlag=ckptFlag
    )

    if cfg.DETECTION.ENABLE:
        """
        预测的时候不执行这一步
        """
        # Load object detector from detectron2
        dtron2_cfg_file = cfg.DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_CFG
        dtron2_cfg = get_cfg()
        dtron2_cfg.merge_from_file(model_zoo.get_config_file(dtron2_cfg_file))
        dtron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
        dtron2_cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_WEIGHTS
        object_predictor = DefaultPredictor(dtron2_cfg)
        # Load the labels of AVA dataset
        with open(cfg.DEMO.LABEL_FILE_PATH) as f:
            labels = f.read().split('\n')[:-1]
        palette = np.random.randint(64, 128, (len(labels), 3)).tolist()
        boxes = []
    else:
        # Load the labels of Kinectics-400 dataset
        labels_df = pd.read_csv(cfg.DEMO.LABEL_FILE_PATH)
        labels = labels_df['name'].values  # ['working' 'sleeping' 'looking_at_phone']
    # frame_provider = VideoReader(cfg)
    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE  # fps = 8
    pred_labels = []
    s = 0.
    frames_nopro = []
    num_positions = len(frame_sits[0])  # num_position = 2

    frame_sits_pro = []
    for frame_1group in frame_sits:
        frames = []
        for i in range(num_positions):
            frame_processed = cv2.cvtColor(frame_1group[i], cv2.COLOR_BGR2RGB)
            frame_processed = scale(cfg.DATA.TEST_CROP_SIZE, frame_processed)
            frames.append(frame_processed)
        frame_sits_pro.append(frames)
    pred_res_str = []
    # frame_sits_pro 就是进行变换之后的 frames_sits,　大小长度都相同
    if len(frame_sits_pro) == seq_len:
        start = time()
        frames = []
        # Pro according to positions
        # print("=" * 80)
        for i in range(num_positions):
            for frame_1group in frame_sits_pro:
                frames.append(frame_1group[i])

            inputs = torch.as_tensor(frames).float()
            inputs = inputs / 255.0  # 归一化
            # Perform color normalization.
            inputs = inputs - torch.tensor(cfg.DATA.MEAN)
            inputs = inputs / torch.tensor(cfg.DATA.STD)
            # [FPS, H, W, C] -> [C, FPS, H, W]
            inputs = inputs.permute(3, 0, 1, 2)
            # 1 C T H W.
            inputs = inputs.unsqueeze(0)

            # 1. Sample frames for the fast pathway.
            index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()

            # [1, 3, 8, h, w]
            fast_pathway = torch.index_select(inputs, 2, index)
            # logger.info('fast_pathway.shape={}'.format(fast_pathway.shape))

            # 2. Sample frames for the slow pathway. cfg.SLOWFAST.ALPHA = 4;
            index = torch.linspace(0, fast_pathway.shape[2] - 1, fast_pathway.shape[2]//cfg.SLOWFAST.ALPHA).long()

            # [1, 3, 4, h, w]
            slow_pathway = torch.index_select(fast_pathway, 2, index)

            # inputs = [slow_pathway, fast_pathway] =[[1, 3, 4, h, w], [1, 3, 8, h, w]]
            inputs = [slow_pathway, fast_pathway]
            # TODO
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for j in range(len(inputs)):
                    inputs[j] = inputs[j].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Perform the forward pass.
            if cfg.DETECTION.ENABLE:
                print("forbiden detection mode!")
                return
            else:
                inputs2 = inputs.copy()
                inputs3 = inputs.copy()
                # print('the first input is  is: ', inputs[0].shape)
                preds1 = model1(inputs)  # 预测部分 shape = [1, 3], 分别是类别的概率
                # print('the second input  is: ', inputs[0].shape)
                preds2 = model2(inputs2)
                preds3 = model3(inputs3)
            # todo work 0.45, sleep 0.2, look phone 0.35
            print('position {}'.format(i+1), "model1 weights1 : ", preds1)
            print('position {}'.format(i+1), "model2 weights2 : ", preds2)
            print('position {}'.format(i+1), "model3 weights3 : ", preds3)

            # print(preds2.detach().cpu().numpy())
            # print("the total result is: ", preds1 + preds2 + preds3)
            # print("-"*60)
            # preds = preds1 + preds2
            label1 = preds1.argmax(-1).cpu().numpy()[0]
            label2 = preds2.argmax(-1).cpu().numpy()[0]

            # 如果有睡觉的情况进行不进行加权判断
            if label1 == 1 or label2 == 1:
                labelIndex1 = preds1.detach().cpu().numpy()
                labelIndex2 = preds2.detach().cpu().numpy()

                # 睡觉的情况，那么当前预测的值还要大于工作或者是看手机0.3以上，才可以。
                if label1 == 1 and (labelIndex1[0][1] - labelIndex1[0][2] > 0.3 and
                                    labelIndex1[0][1] - labelIndex1[0][0] > 0.3):
                    preds = preds1

                elif label2 == 1 and (labelIndex2[0][1] - labelIndex2[0][2] > 0.3 and
                                    labelIndex2[0][1] - labelIndex2[0][0] > 0.3):
                    preds = preds2
                # 其他情况直接加权即可
                else:
                    preds = preds1 + preds2 + preds3
                    preds[0][0] = preds[0][0] * 0.45
                    preds[0][1] = preds[0][1] * 0.2
                    preds[0][2] = preds[0][2] * 0.35

            # 工作或者是看手机情况
            else:
                if weightsSum:
                    preds = preds1 + preds2 + preds3
                    preds[0][0] = preds[0][0] * 0.45
                    preds[0][1] = preds[0][1] * 0.2
                    preds[0][2] = preds[0][2] * 0.35

            # print('after weights ', preds)
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds = du.all_gather(preds)[0]
                print("using gpu model")

            if cfg.DETECTION.ENABLE:
                print("forbiden detection mode!")
                return
            else:
                if weightsSum:
                    label_id = preds.argmax(-1).cpu()
                    pred_labels = labels[label_id]
                else:
                    label = []
                    label1 = preds1.argmax(-1).cpu().numpy()[0]
                    label2 = preds2.argmax(-1).cpu().numpy()[0]
                    label3 = preds3.argmax(-1).cpu().numpy()[0]
                    print('the label is: ', label1, label2, label3)
                    label.insert(0, label1)
                    label.insert(0, label2)
                    label.insert(0, label3)
                    labelid = max(label, key=label.count)
                    pred_labels = labels[labelid]

                # label_ids = preds.argmax(-1).cpu().detach().numpy()
                # pred_labels = labels[label_ids]
                # Option 2: multi-label inferencing selected from probability entries > threshold
                # label_ids = torch.nonzero(preds.squeeze() > .1).reshape(-1).cpu().detach().numpy()
                # pred_labels = labels[label_ids]
                # logger.info(pred_labels)
                if not list(pred_labels):
                    pred_labels = ['Unknown']

            # add move_recognition results to human_exists, Not work! ???
            # human_exists[i][1].update({'move' : pred_labels})
            # human_exists[i][1]['status'] = pred_labels
            pred_res_str.append(pred_labels)

            # # option 1: remove the oldest frame in the buffer to make place for the new one.
            # frames.pop(0)
            # option 2: empty the buffer
            frames = []
            # s = time() - s1
            # logger.info(pred_labels)
            # logger.info('end detect --cost time is:-->'+str(s))
    return pred_res_str




