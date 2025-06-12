from sklearn import tree, ensemble

models = {
    # 以gini系数度量的决策树
    "decision_tree_gini": tree.DecisionTreeClassifier(
        criterion="gini"
    ),
    # 以entropy系数度量的决策树
    "decision_tree_entropy": tree.DecisionTreeClassifier(
        criterion="entropy"
    ),
 # 随机森林模型
    "rf": ensemble.RandomForestClassifier(),
}