import numpy as np
from mxnet import ndarray
from sklearn.metrics import mean_squared_error

np.seterr(invalid='ignore')
# 需要测量角度的部位，每个部位需要用它本身和与之连接的两个节点来计算角度
# 第一点是关节点
KeyPoints = [
    (5, 6, 7),       # 左肩
    (6, 7, 8),       # 右肩
    (7, 5, 9),       # 左臂
    (8, 6, 10),      # 右臂
    (11, 5, 13),     # 左胯
    (12, 6, 14),     # 右胯
    (13, 11, 15),    # 左膝
    (14, 12, 16),    # 右膝
]

class AngeleCal():

    def __init__(self, filename):
        self.stdAngles = np.loadtxt(filename, delimiter='\t')
        self.pos = 0

    # 计算角度
    @staticmethod
    def cal(coords, confidence, keypoint_thresh=0.2):
        joint_visible = confidence[:, :, 0] > keypoint_thresh
        angles = np.empty((coords.shape[0], len(KeyPoints)))

        for i, pts in enumerate(coords):
            # 某个人
            for j, keyPoint in enumerate(KeyPoints):
                # 是否识别到这个关节
                if joint_visible[i, keyPoint[0]] and joint_visible[i, keyPoint[1]] and joint_visible[i, keyPoint[2]]:
                    # 坐标
                    p0x = pts[keyPoint[0], 0].asscalar()
                    p0y = pts[keyPoint[0], 1].asscalar()
                    p1x = pts[keyPoint[1], 0].asscalar()
                    p1y = pts[keyPoint[1], 1].asscalar()
                    p2x = pts[keyPoint[2], 0].asscalar()
                    p2y = pts[keyPoint[2], 1].asscalar()

                    # 角度
                    angle1 = np.arctan2(p1y - p0y, p1x - p0x)
                    angle2 = np.arctan2(p2y - p0y, p2x - p0x)
                    angles[i][j] = angle2 - angle1
                else:
                    angles[i][j] = np.nan
    
        return angles

    # 角度对比
    def compare(self, angles):
        stdAngle = self.stdAngles[self.pos]     # 每次读取一行标准角度
        scores = np.empty(len(angles))
        visibles = ~np.isnan(stdAngle)          # 样本中没有缺失值的
        stdAngle = stdAngle[visibles]           # 过滤掉样本中的缺失值
        for i, angle in enumerate(angles):
            angle_v = angle[visibles]           # 过滤样本中也有缺失值的点
            if np.isnan(angle_v).any():         # 还有缺失值
                scores[i] = np.nan
            else:
                scores[i] = mean_squared_error(angle_v, stdAngle)
        self.pos += 1

        return 100 * (1 - np.tanh(scores))