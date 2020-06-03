import argparse, os
import cv2 as cv
import mxnet as mx
import numpy as np
from gluoncv.data.transforms.presets.ssd import transform_test
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints
from mxnet.gluon.data.vision import transforms
from model import ctx, detector, estimator
from fps import FPS
from angle import AngeleCal

# 读取参数
parser = argparse.ArgumentParser()
parser.add_argument('--input', default=0)
parser.add_argument('--demo', required=True)
parser.add_argument('--data', required=True)
args = parser.parse_args()

# 视频读取
# 1是输入视频，2是示例视频
cap1 = cv.VideoCapture(args.input)
cap2 = cv.VideoCapture(args.demo)

# 标准特征
angeleCal = AngeleCal(args.data)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

while ret1 and ret2:

    frame1 = mx.nd.array(cv.cvtColor(frame1, cv.COLOR_BGR2RGB)).astype('uint8')
    frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB) 

    # 目标检测
    x, img = transform_test(frame1, short=512)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, ctx=ctx)

    # 姿态识别
    if len(upscale_bbox) > 0:
        predicted_heatmap = estimator(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        img = cv_plot_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxs, scores)

        # 动作对比
        angles = AngeleCal.cal(pred_coords, confidence)
        results = angeleCal.compare(angles)
    else:
        results = ['NaN']

    # 缩放示例视频并合并显示
    height = int(img.shape[0])
    width = int(height * frame2.shape[1] / frame2.shape[0])
    frame2 = cv.resize(frame2, (width, height))
    img = np.hstack((img, frame2))

    cv_plot_image(img, 
        upperleft_txt=FPS.fps(), upperleft_txt_corner=(10,25),
        left_txt_list=results)
    
    # ESC键退出
    if cv.waitKey(1) == 27:
        break

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

cv.destroyAllWindows()

cap1.release()
cap2.release()