import argparse
import os
import model_dispatcher
import joblib
import config
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold,model):
    # 读取数据文件
    df = pd.read_csv(config.RAINING_FILE)
    # 选取df中kfold列不等于fold
    df_train = df[df.kfold!= fold].reset_index(drop=True)
    # 选取df中kfold列等于fold
    df_valid = df[df.kfold==fold].reset_index(drop=True)

    # 训练集输入，删除label列
    x_train = df_train.drop("label", axis=1).values
    # 训练集输出，取label列
    y_train = df_train.label.values
    # 验证集输入，删除label列
    x_valid = df_valid.drop("label", axis=1).values
    # 验证集输出，取label列
    y_valid = df_valid.label.values
    # 实例化决策树模型
    clf = model_dispatcher.models[model]
    # 使用训练集训练模型
    clf.fit(x_train, y_train)
    # 使用验证集输入得到预测结果
    preds = clf.predict(x_valid)
    # 计算验证集准确率
    accuracy = metrics.accuracy_score(y_valid, preds)
    # 打印fold信息和准确率
    print(f"Fold={fold}, Accuracy={accuracy}")
    # 保存模型
    joblib.dump(clf,os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin") )

if __name__ == "__main__":
    # 运行每个折叠
    # 实例化参数环境
    parser = argparse.ArgumentParser()
    # fold参数
    parser.add_argument("--fold", type=int)
    # 读取参数
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    run(fold=args.fold,model=args.model)