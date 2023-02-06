# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="/home/bruce/PycharmProjects/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=[],
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='/home/bruce/bigVolumn/autolabelData/detectron7',
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        # default=['MODEL.WEIGHTS', 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'],
        # default=['MODEL.WEIGHTS', '/home/bruce/PycharmProjects/detectron2/weights/model_final_f10217.pkl'],
        default=['MODEL.WEIGHTS', "/home/bruce/PycharmProjects/detectron2/weights/model_final_280758.pkl"],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    # imgPath = "/home/bruce/PycharmProjects/detectron2/inputs"
    # imgPath = "/home/bruce/PycharmProjects/keras_ocr/autolabel/imgSimilar"
    imgPath = "/home/bruce/bigVolumn/autolabelData/testVideoData"
    args.input = [os.path.join(imgPath, i) for i in os.listdir(imgPath)]
    if os.path.exists(args.output):
        pass
    else:
        os.mkdir(args.output)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            # print('the img shape is:', img.shape)
            predictions, visualized_output = demo.run_on_image(img)
            # print('the prediction is: ', predictions)
            # print('the instance is:', predictions['instances'].pred_boxes.tensor.cpu().clone().numpy())  # 在boxes的类里面有tensor属性,可以直接利用
            # print('the predition scores is:', predictions['instances'].scores.cpu().clone().numpy())
            # print('\n the predictions class is: \n', predictions['instances'].pred_classes.cpu().clone().numpy())
            # print('the visualized_output box coordinate is: \n', visualized_output)
            # print('visualized output is: ', visualized_output.get_image())
            # show the result：
            # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # 调整图像
            # cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            # if cv2.waitKey(0) == 'q':
            #     continue  # esc to quit

            # 只选择 0 person 方框和标签
            boxes_ = predictions['instances'].pred_boxes.tensor.cpu().clone().numpy()
            scores_ = predictions['instances'].scores.cpu().clone().numpy()
            classes_ = predictions['instances'].pred_classes.cpu().clone().numpy()
            index = [i for i in range(classes_.shape[0]) if classes_[i] == 0]
            boxes_ = boxes_[index]
            scores_ = scores_[index]
            classes_ = classes_[index]
            # print(boxes_, classes_, scores_)
            # print(time.time())
            print("consume time: ", time.time() - start_time)

            # 打印中间信息
            # logger.info(
            #     "{}: {} in {:.2f}s".format(
            #         path,
            #         "detected {} instances".format(len(predictions["instances"]))
            #         if "instances" in predictions
            #         else "finished",
            #         time.time() - start_time,))
            # for i in list(predictions['instances'].defdict()['pred_classes']): # 得到label
                # print('the ith info is:', i.cpu().numpy())

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                print("show output file ")
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # 调整图像
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
