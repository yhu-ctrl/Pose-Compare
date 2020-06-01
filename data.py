import argparse, time, os
import cv2 as cv
import mxnet as mx
import numpy as np
import gluoncv as gcv
from gluoncv import data
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints
from mxnet.gluon.data.vision import transforms
from angle import AngeleCal

# 读取参数
parser = argparse.ArgumentParser()
parser.add_argument('--video')
args = parser.parse_args()

fps_time = 0

# 设置模型
ctx = mx.gpu()

detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)

estimator_name = "simple_pose_resnet18_v1b"
estimator = get_model(estimator_name, pretrained='ccd24037', ctx=ctx)

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})

detector.hybridize()
estimator.hybridize()

# 视频读取
cap = cv.VideoCapture(args.video)

ret, frame = cap.read()
features = []
while ret:

    # 目标检测
    frame = mx.nd.array(cv.cvtColor(frame, cv.COLOR_BGR2RGB)).astype('uint8')

    x, img = gcv.data.transforms.presets.ssd.transform_test(frame, short=512)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, output_shape=(128, 96), ctx=ctx)

    # 只识别一个人的姿态
    if len(upscale_bbox) == 1:
        predicted_heatmap = estimator(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        img = cv_plot_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxs, scores)

        X = AngeleCal.cal(pred_coords, confidence)[0]
        print(X)
        features.append(X)
    else:
        # 人数不对就插入nan
        print(np.nan)
        features.append(np.nan)

    ret, frame = cap.read()

cap.release()

# 将一个视频的特征保存到文件
np.savetxt(os.path.join('data', 'demo.tsv'), np.array(features), delimiter='\t', fmt='%4f')