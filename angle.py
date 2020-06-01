import numpy as np
from mxnet import ndarray
from sklearn.metrics import r2_score


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

    # 角度对比
    def compare(self, angles):
        # 每次读取一行标准角度
        stdAngle = self.stdAngles[self.pos]
        scores = []
        visibles = ~np.isnan(stdAngle)          # 样本中没有缺失值的
        for angle in angles:
            angle_v = angle[visibles]           # 过滤样本中也有缺失值的点
            if np.isnan(angle_v).any():         # 还有缺失值
                scores.append('NaN')
            else:
                scores.append('{:.4f}'.format(r2_score(angle_v, stdAngle[visibles])))
        self.pos += 1

        return scores