"""   
@Project Name: ML
@Author: Shen Hongcai
@Time: 2019-03-29, 13:24
@Python Version: python3.6
@Coding Scheme: utf-8
@Interpreter Name: PyCharm
"""

import numpy as np
import cv2

box = [[635, 111, 849, 375, 0.92],
       [658, 131, 891, 426, 0.88],
       [618, 104, 865, 393, 0.95],
       [904, 9, 1156, 328, 0.90],
       [921, 23, 1148, 346, 0.85]]


def nms(boxes, thresh=0.3):
    boxes=np.asarray(boxes)
    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]
    scores = boxes[:, 4]  # bbox置信度
    areas = (xmax - xmin + 1) * (ymax - ymin + 1)  # 每一个检测框的面积,array形式
    order = scores.argsort()[::-1]  # 置信度降序排列，返回降序后的box索引

    # keep为最后保留的边框的索引
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)

        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(xmin[i], xmin[order[1:]])
        yy1 = np.maximum(ymin[i], ymin[order[1:]])
        xx2 = np.minimum(xmax[i], xmax[order[1:]])
        yy2 = np.minimum(ymax[i], ymax[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = w * h
        union = areas[i] + areas[order[1:]] - overlap
        IOU = overlap / union
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(IOU <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep


def plotRectangle(boxes):
    """
    :param img: narray
    :param location: [xmin,ymin,xmax,ymax]
    :return:
    """
    img = cv2.imread("/Users/wangjiaxin/Desktop/workspace2/ExercisePython/ML/nms/IMG_3600.PNG")

    xmin = [boxes[i][0] for i in range(len(boxes))]
    ymin = [boxes[i][1] for i in range(len(boxes))]
    xmax = [boxes[i][2] for i in range(len(boxes))]
    ymax = [boxes[i][3] for i in range(len(boxes))]
    strs=["0.95","0.9"]

    linecolor = (0, 255, 0)
    linetype = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(boxes)):
        cv2.rectangle(img, (xmin[i],ymin[i]), (xmax[i],ymax[i]), linecolor, linetype)
        cv2.putText(img,strs[i], (xmin[i]+5,ymax[i]), font, 1.2, (255, 0, 0), 2)
    cv2.waitKey(0)
    cv2.imwrite("./result_labelFace1.jpg", img)  # 存储结果图像


b =[box[i][0:4] for i in [2, 3]]
plotRectangle(b)