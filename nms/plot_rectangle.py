"""   
@Project Name: ML
@Author: Shen Hongcai
@Time: 2019-03-29, 17:31
@Python Version: python3.6
@Coding Scheme: utf-8
@Interpreter Name: PyCharm
"""
import cv2


def plotRectangle(img, boxes, information):

    """
    :param img: ndarray
    :param boxes: bbox location list  [xmin,ymin,xmax,ymax]
    :param information: label list  for rectangle
    :return:
    """
    xmin = [boxes[i][0] for i in range(len(boxes))]
    ymin = [boxes[i][1] for i in range(len(boxes))]
    xmax = [boxes[i][2] for i in range(len(boxes))]
    ymax = [boxes[i][3] for i in range(len(boxes))]
    label = information
    linecolor = (0, 255, 0)   # green
    linetype = 2            # line width
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontsize = 0.8
    fontcolor=(0, 0, 255)

    for i in range(len(boxes)):
        cv2.rectangle(img, (xmin[i], ymin[i]), (xmax[i], ymax[i]), linecolor, linetype)
        cv2.putText(img, label[i], (xmin[i]+5, ymax[i]), font, fontsize, fontcolor, 2)

    cv2.waitKey(0)
    cv2.imwrite("plotpicture.png",img)



if __name__=="__main__":
    imgpath="/Users/wangjiaxin/Desktop/workspace2/ExercisePython/ML/nms/IMG_3600.PNG"
    image=cv2.imread(imgpath)
    box = [[635, 111, 849, 375, 0.92],
           [658, 131, 891, 426, 0.88],
           [618, 104, 865, 393, 0.95],
           [904, 9, 1156, 328, 0.90],
           [921, 23, 1148, 346, 0.85]]
    keep=[i for i in range(len(box))]
    keepbox = [box[i][0:4] for i in keep]
    score = [str(box[i][-1]) for i in keep]
    plotRectangle(image,keepbox,score)









