"""   
@Project Name: ML
@Author: Shen Hongcai
@Time: 2019-03-29, 16:54
@Python Version: python3.6
@Coding Scheme: utf-8
@Interpreter Name: PyCharm
"""

import cv2
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










