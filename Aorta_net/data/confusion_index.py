import numpy as np


def conf_index(confusion_matrix):
    # 2-TP/TN/FP/FN的计算
    weight=confusion_matrix.sum(axis=0)/confusion_matrix.sum()## 求出每列元素的和
    FN = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)#所有对的 TP.sum=TP+TN
    TN = confusion_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # print(TP)
    # print(TN)
    # print(FP)
    # print(FN)

    # 3-其他的性能参数的计算
    TPR = TP / (TP + FN)  # Sensitivity/ hit rate/ recall/ true positive rate 对于ground truth
    TNR = TN / (TN + FP)  # Specificity/ true negative rate  对于
    PPV = TP / (TP + FP)  # Precision/ positive predictive value  对于预测而言
    NPV = TN / (TN + FN)  # Negative predictive value
    FPR = FP / (FP + TN)  # Fall out/ false positive rate
    FNR = FN / (TP + FN)  # False negative rate
    FDR = FP / (TP + FP)  # False discovery rate
    sub_ACC = TP / (TP + FN)  # accuracy of each class
    acc=(TP+TN).sum()/(TP+TN+FP+FN).sum()
    average_acc=TP.sum() / (TP.sum() + FN.sum())
    F1_Score=2*TPR*PPV/(PPV+TPR)
    Macro_F1=F1_Score.mean()
    weight_F1=(F1_Score*weight).sum()# 应该把不同类别给与相同权重，不应该按照数量进行加权把？
    print('acc:',average_acc)
    print('Sensitivity:', TPR.mean())#Macro-average方法
    print('Specificity:', TNR.mean())
    print('Macro_F1:',Macro_F1)

def ca1():
    # 1-混淆矩阵
    matrix1 = np.array(
        [[9, 0, 0,0],
         [0, 2, 0,0],
         [0, 0, 2,0],
         [0, 0, 0,27]])

    # 1-混淆矩阵
    matrix2 = np.array(
        [[24, 0, 0,0],
         [0, 1, 0,0],
         [0, 0, 15,1],
         [0, 0, 0,144]])

    matrix3 = np.array(
        [[17, 0, 0,0],
         [0, 5, 0,1],
         [0, 0, 24,2],
         [0, 1, 1,217]])

    # 1-混淆矩阵 行为predict，列为groundtruth
    matrix = np.array(
        [[50, 0, 0,0],
         [0, 8, 0,1],
         [0, 0, 41,3],
         [0, 1, 1,388]])
    conf_index(matrix)
if __name__ == '__main__':
    ca1()

