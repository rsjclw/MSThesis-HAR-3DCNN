#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from argparse import Namespace
import os
import time
# from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import numpy as np

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

class Predictor(object):
    def __init__(self, device='gpu'):
        args = Namespace(
            camid = 0, ckpt = 'checkpoints/yolox_tiny.pth', conf = 0.7, device = device,
            exp_file = None, experiment_name = None, fp16 = False, fuse = False, legacy = False,
            name = 'yolox-tiny', nms = 0.5, save_result = True, trt = False, tsize = 640
        )

        exp = get_exp(None, args.name)
        if not args.experiment_name:
            args.experiment_name = exp.exp_name

        file_name = os.path.join(exp.output_dir, args.experiment_name)
        os.makedirs(file_name, exist_ok=True)

        vis_folder = None
        if args.save_result:
            vis_folder = os.path.join(file_name, "vis_res")
            os.makedirs(vis_folder, exist_ok=True)

        if args.trt:
            args.device = "gpu"

        # logger.info("Args: {}".format(args))

        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)

        model = exp.get_model()
        # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

        if args.device == "gpu":
            model.cuda()
            if args.fp16:
                model.half()  # to FP16
        model.eval()

        if not args.trt:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            # logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            # logger.info("loaded checkpoint done.")

        if args.fuse:
            # logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.trt:
            assert not args.fuse, "TensorRT model is not support model fusing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            # logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None
        self.model = model
        self.cls_names = COCO_CLASSES
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = args.device
        self.fp16 = args.fp16
        self.preproc = ValTransform(legacy=args.legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        img_info["file_name"] = ""

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35, getBestOnly=False):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return None, None
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        return vis(img, bboxes, scores, cls, cls_conf, self.cls_names, getBestOnly)

    def detect(self, img, getBestOnly=False):
        outputs, img_info = self.inference(img)
        cropped_frame, bboxes = self.visual(outputs[0], img_info, self.confthre, getBestOnly)
        # cv2.imwrite(str(time.time())+'.jpg', result_image)#cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        return cropped_frame, bboxes
        # cv2.imwrite(str(time.time())+'C.jpg', cropped_frame)


# def main():
#     img_path = 'assets/1671958948.8045166.jpg'
#     vid_path = 'sop_dataset/back_head/0300049.mp4'
#     cap = cv2.VideoCapture(vid_path)
#     predictor = Predictor()
#     ret, frame = cap.read()
#     while ret:
#         # img, box = predictor.detect(cv2.imread('chizuru.jpg'))
#         img, box = predictor.detect(frame)
#         print(frame.shape, box)
#         ret, frame = cap.read()
#         # if box is not None:
#         #     print(frame.shape, box.shape)
#         # else:
#         #     print(frame.shape, box)

# if __name__ == "__main__":
#     main()
