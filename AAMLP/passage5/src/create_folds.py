import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../../../dataset/passage5/input/mnist_train.csv")
    print(df.shape)

    df["kfold"] = -1  # 创建一个名为 kfold 的新列，并用-1填充
    df = df.sample(frac=1).reset_index(drop=True)  # 打乱数据

    kf = model_selection.KFold(n_splits=5)  # 实例化（5折交叉验证）

    for fold, (trn_, val_) in enumerate(kf.split(X=df)):  # 填充新的 kfold 列
        df.loc[val_, 'kfold'] = fold
        print(fold, trn_, val_)


    df.to_csv("../input/mnist_train_folds.csv", index=False)  # 保存划分好的数据集





