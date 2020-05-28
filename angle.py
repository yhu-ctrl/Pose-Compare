import numpy as np
from mxnet import ndarray

# 需要测量角度的部位，每个部位需要用它本身和与之连接的两个节点来计算角度
# 第一点是关节点
KeyPoints = [
    (5, 6, 7),       # 左肩
    (6, 7, 8),       # 右肩
    (7, 5, 9),      # 左臂
    (8, 6, 10),      # 右臂
    (11, 5, 13),     # 左胯
    (12, 6, 14),     # 右胯
    (13, 11, 15),    # 左膝
    (14, 12, 16),    # 右膝
]

# 计算所有人关键部位的夹角的余弦值
def CalAngle(coords, confidence, keypoint_thresh=0.2):
    joint_visible = confidence[:, :, 0] > keypoint_thresh
    angles = np.empty((coords.shape[0], len(KeyPoints)))

    for i, pts in enumerate(coords):
        # 某个人
        for j, keyPoint in enumerate(KeyPoints):
            # 是否识别到这个关节
            if joint_visible[i, keyPoint[0]] and joint_visible[i, keyPoint[1]] and joint_visible[i, keyPoint[2]]:
                # 计算
                # print(pts)

                p0x = pts[keyPoint[0], 0].asscalar()
                p0y = pts[keyPoint[0], 1].asscalar()
                p1x = pts[keyPoint[1], 0].asscalar()
                p1y = pts[keyPoint[1], 1].asscalar()
                p2x = pts[keyPoint[2], 0].asscalar()
                p2y = pts[keyPoint[2], 1].asscalar()

                v1 = np.array([ p1x - p0x, p1y - p0y ])
                v2 = np.array([ p2x - p0x, p2y - p0y ])

                angles[i][j] = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

            else:
                angles[i][j] = np.nan
   
    return angles