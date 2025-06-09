# 导入 pandas 和 scikit-learn 的 model_selection 模块
import pandas as pd
from sklearn import model_selection


# 训练数据存储在名为 train.csv 的 CSV 文件中
df = pd.read_csv("../dataset/archive/winequality-red.csv")
# 我们创建一个名为 kfold 的新列，并用 -1 填充
df["kfold"] = -1
# 接下来的步骤是随机打乱数据的行
df = df.sample(frac=1).reset_index(drop=True)
# 从 model_selection 模块初始化 kfold 类
kf = model_selection.KFold(n_splits=5)
# 填充新的 kfold 列（enumerate的作用是返回一个迭代器）
for fold, (trn_, val_) in enumerate(kf.split(X=df)):
    df.loc[val_, 'kfold'] = fold
    # 保存带有 kfold 列的新 CSV 文件
df.to_csv("../dataset/cross_check_test/train_folds1.csv", index=False)