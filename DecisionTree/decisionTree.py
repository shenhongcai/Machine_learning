"""   
@Project Name: ML
@Author: Shen Hongcai
@Time: 2019-03-27, 23:18
@Python Version: python3.6
@Coding Scheme: utf-8
@Interpreter Name: PyCharm
"""
import pandas as pd
import numpy as np
def _init_data():
    dataSet1 = [['爱学习', '不打游戏', '沉迷女色', '是'],
                ['爱学习', '打游戏', '不沉迷女色', '是'],
                ['不爱学习', '打游戏', '沉迷女色', '否'],
                ['不爱学习', '打游戏', '不沉迷女色', '否']]
    labels1 = ['是否爱学习', '是否天天打游戏', '是否沉迷女色', '是否是优秀学生']

    dataSet2 = [['长', '粗', '男'],
                ['短', '粗', '男'],
                ['短', '粗', '男'],
                ['长', '细', '女'],
                ['短', '细', '女'],
                ['短', '粗', '女'],
                ['长', '粗', '女'],
                ['长', '粗', '女']]
    labels2 = ['头发', '声音', "性别"]

    dataSet3 = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
                ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
                ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
                ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
                ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
                ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
                ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
                ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
                ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
                ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
                ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
                ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
                ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
                ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
                ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
                ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
                ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']]
    labels3 = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    data1 = pd.DataFrame(dataSet2, columns=labels2)
    data = data1.set_index(labels2[-1])
    return data
data=_init_data()

def calEnt(data):
    num=len(data)
    rootLabel = data.index.value_counts()
    shannoEnt = 0
    for i in range(len(rootLabel)):
        prob = rootLabel[i]/num
        shannoEnt += -prob*np.log2(prob)
    return shannoEnt

indexEnt = calEnt(data)


def selectBestFeature(data,indexEnt):
    LabelData = list(data.index)
    if LabelData.count(LabelData[0]) == len(LabelData):
        return LabelData[0]
    if len(data.columns) == 0:
        return data.index.value_counts().sort_values(ascending=False).index[0]
    if len(data.columns)>0:
        columnslist = data.columns
        featurelist = [f for f in data.columns]
        featuresgainEnt =[]  # 存储各个属性的香农熵增益
        for feature in featurelist:
            featurevalue = set(data[feature].values)
            tempsum=0
            for uniqueValue in featurevalue:
                subdata=data.loc[data[feature]==uniqueValue]
                tempsum += (len(subdata)/len(data))*calEnt(subdata)
            featuresgainEnt.append(indexEnt-tempsum)

        xdict=dict(zip(columnslist,featuresgainEnt))
        print(xdict)
        top = max(xdict, key=xdict.get)
        mytree={top:{}}
        for value in set(data[top]):
            newdata=data[data[top]==value].drop(top,axis=1)
            mytree[top][value]=selectBestFeature(newdata,calEnt(newdata))
        return mytree

ans=selectBestFeature(data,indexEnt)
print(ans)


