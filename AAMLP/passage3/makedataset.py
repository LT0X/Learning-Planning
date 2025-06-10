
# 为回归问题进行分层K-折交叉验证
# 导入需要的库
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection


# 创建分折（folds）的函数
def create_folds(data):
# 创建一个新列叫做kfold，并用-1来填充
    data["kfold"] = -1

    # 随机打乱数据的行

    data = data.sample(frac=1).reset_index(drop=True)

    # 使用Sturge规则计算bin的数量
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # 使用pandas的cut函数进行目标变量（target）的分箱
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )

    # 初始化StratifiedKFold类
    kf = model_selection.StratifiedKFold(n_splits=5)

    # 填充新的kfold列
    # 注意：我们使用的是bins而不是实际的目标变量（target）！
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    # 删除bins列
    data = data.drop("bins", axis=1)

    # 返回包含folds的数据
    return data

# 主程序开始
if  __name__ == "__main__":
    # 创建一个带有15000个样本、100个特征和1个目标变量的样本数据集
    X, y = datasets.make_regression(
        n_samples=15000, n_features=100, n_targets=1
    )

    # 使用numpy数组创建一个数据框
    df = pd.DataFrame(
        X,
        columns=[f"f_{i}" for i in range(X.shape[1])]
    )
    df.loc[:, "target"] = y

    # 创建folds
    df = create_folds(df)
    print(df.head())