使用类实现Adaboost算法以及对数回归和决策树桩
在外部进行测试输入如下代码：
test1 = Adaboost(base = 0)
test1.fit("data.csv", "targets.csv")
data = test1.predict("datas.csv")
便会进行十折交叉验证将数据写入目标文件并且输出正确率，
返回data为对datas数据进行预测的结果。
测试时data.csv,targets.csv,datas.csv要与程序放在同一文件夹下！