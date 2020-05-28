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
from angle import CalAngle
from sklearn.metrics import r2_score

# 读取参数
parser = argparse.ArgumentParser()
parser.add_argument('--input', default=0)
parser.add_argument('--demo', required=True)
parser.add_argument('--data', required=True)
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
cap1 = cv.VideoCapture(args.input)
cap2 = cv.VideoCapture(args.demo)

# 标准特征
stdAngle = np.loadtxt(args.data, delimiter='\t')
pos = 0

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()
while ret1 and ret2:

    # 目标检测
    frame = mx.nd.array(cv.cvtColor(frame1, cv.COLOR_BGR2RGB)).astype('uint8')

    x, img = gcv.data.transforms.presets.ssd.transform_test(frame, short=512)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, output_shape=(128, 96), ctx=ctx)

    # 姿态识别
    if len(upscale_bbox) > 0:
        predicted_heatmap = estimator(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        img = cv_plot_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxs, scores)

        # 动作对比
        scores = []
        # print(stdAngle[pos])
        visibles = ~np.isnan(stdAngle[pos])     # 样本中没有缺失值的点
        angles = CalAngle(pred_coords, confidence)
        for angle in angles:
            angle_v = angle[visibles]           # 过滤样本中也有缺失值的点
            print(angle_v)
            if np.isnan(angle_v).any():         # 还有缺失值
                scores.append('NaN')
            else:
                scores.append('{:.4f}'.format(r2_score(angle_v, stdAngle[pos][visibles])))
        pos += 1

    cv_plot_image(img, 
        upperleft_txt=f"FPS:{(1.0 / (time.time() - fps_time)):.2f}", upperleft_txt_corner=(10,25),
        left_txt_list=scores, canvas_name='pose')
    fps_time = time.time()
    # cv.imshow('demo', frame2)
    
    # ESC键退出
    if cv.waitKey(1) == 27:
        break

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

cv.destroyAllWindows()

cap1.release()
cap2.release()