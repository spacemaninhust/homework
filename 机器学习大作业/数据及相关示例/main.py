
import numpy as np
import os
import math
#读文件函数
def loadData(x_file, y_file):
    current_path = os.path.dirname(__file__)
    data = open(current_path + '/' + x_file)
    data_2 = open(current_path + '/' + y_file)
    dataMat = []
    labelMat = []
    for dataline in data.readlines():
        data_3 = []
        linedata = dataline.strip('\n')
        linedata = dataline.split(',')
        for i in linedata:
            data_3.append(float(i))
        dataMat.append(data_3)
    for dataline in data_2.readlines():
        labelMat.append(float(dataline[0]))
    return dataMat,labelMat
#计算相关系数（皮尔斯系数函数）
def pearson(v1, v2):
    n = len(v1)
    #simple sums
    sum1 = sum((v1[i]) for i in range(n))
    sum2 = sum((v2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2) for v in v1])
    sum2_pow = sum([pow(v, 2) for v in v2])
    #sum up the products
    p_sum = sum([v1[i] * v2[i] for i in range(n)])
    #分子num，分母denominator
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den
#归一化函数
def autoNorm(data): #传入一个矩阵
    mins = data.min(0) #返回data矩阵中每一列中最小的元素，返回一个列表
    maxs = data.max(0) #返回data矩阵中每一列中最大的元素，返回一个列
    ranges = maxs - mins #最大值列表 - 最小值列表 = 差值列表
    normData = np.zeros(np.shape(data)) #生成一个与 data矩阵同规格的normData全0矩阵，
                                        #用于装归一化后的数据
    row = data.shape[0] #返回 data矩阵的行数
    normData = data - np.tile(mins,(row,1)) #data矩阵每一列数据都减去每一列的最小值
    normData = normData / np.tile(ranges,(row,1)) #data矩阵每一列数据都除去每一列的差值
                                                    #(差值 = 某列的最大值- 某列最小值)
    return normData
#进行数据预处理函数，选择57个特征属性中相关性最高的30个
def process(x_file, y_file):
    x,y = loadData(x_file, y_file)
    x = np.mat(x)
    y = np.array(y)
    #x = autoNorm(x)
    p = []
    for i in range(x.shape[1]):
        pccs = pearson(x[:,i],y)
        p.append([pccs,i])
    t = sorted(p,key= lambda x:x[0])
    index = []
    for i in range(1,31):
        index.append(t[-i][1])
    data = []
    for i in index:
        data.append(x[:,i])
    data = np.array(data).T
    data = data.reshape(data.shape[1],data.shape[2])
    return data, y
#读取目标文件
def loaddatas(x_file):
    current_path = os.path.dirname(__file__)
    data = open(current_path + '/' + x_file)
    dataMat = []
    for dataline in data.readlines():
        data_3 = []
        linedata = dataline.strip('\n')
        linedata = dataline.split(',')
        for i in linedata:
            data_3.append(float(i))
        dataMat.append(data_3)
    return dataMat
#计算准确率
def get_acc(y,x):
    acc = np.sum(y == x) / len(y)
    return acc
#正则化对数
class Logistic():
    def __init__(self):
        self.alpha = 0.001
        self.iter_num = 20000
        self.lamda = 0

    def model(self,X,theta):
        z = np.dot(X,theta)
        h = 1/(1 + np.exp(-z))
        return h

    # 定义梯度下降
    def gradeDesc(self,X,y):
        # 初始化工作
        # 获取数据的维度
        m,n = X.shape
        # 初始化theta
        theta = np.zeros((n,1))
        # 初始化代价记录列表
        #执行梯度下降
        for i in range(self.iter_num):
            # 获取在当前theta下的数据的预测值
            h = self.model(X,theta)
            # 定义正则化项
            # 复制一个theta,使theta[0] 为0
            theta_r = theta.copy()
            theta_r[0] = 0
            deltaTheta = (1/m) * (np.dot(X.T,(h - y)) + self.lamda* theta_r)
            # 更新theta
            theta -= self.alpha * deltaTheta 
        # 模型训练完毕，返回训练好的theta
        return theta
    def predict(self,dataset,w):
        predictdata = self.model(dataset,w)
        for i in range(len(predictdata)):
            if predictdata[i] > 0.5:#预测结果大于0.5，则分为1类
                predictdata[i] = 1
            else:
                predictdata[i] = 0#预测结果小于0.5，则分为0类
        return predictdata
#对数几率类
class logical():
    def __init__(self):
        self.numIter = 100

    def sigmoid(self, inX):   # 定义sigmoid函数
        return 1.0/(1+np.exp(-inX))
    
    def stocGradAscent1(self, dataMatrix, classLabels, D):#因为由图可知迭代到150次左右就收敛（达到稳定值）
        m,n = np.shape(dataMatrix)   # 取数组（narray）的行，列 
        weights = np.ones(n)  # [1. 1. 1.]
        minerror = float("inf")
        for j in range(self.numIter):  # 循环到 最大循环次数numIter = 60：
            dataIndex = list(range(m))  
            for i in range(m):   # 循环listIndex
                alpha = 4/(1.0+j+i)+0.01#动态调整步进因子
                randIndex = int(np.random.uniform(0, len(dataIndex)))   # 随机选取样本来更新回归系数
                h = self.sigmoid(sum(dataMatrix[randIndex]*weights))   #  1.0/(1+exp(-Z))    Z = dataMatrix * weights
                error = classLabels[randIndex] - h
                weights = weights + alpha * error * dataMatrix[randIndex]#更新权重
                del(dataIndex[randIndex])   # 删除用过的 随机数 （避免重复）  
            predict_datass = self.predict(np.mat(dataMatrix),np.mat(weights).T)
            errArr = np.mat(np.ones((m,1)))#创建错误向量，初始值为1
            labelsets = np.mat(classLabels).T
            errArr[predict_datass == labelsets] = 0#若预测的值和真实值相同，则赋值为0
            weighterror = D.T * errArr#权重与错误向量相乘求和
            if  weighterror < minerror:
                minerror = weighterror
                bestdata = predict_datass.copy()
                w = weights
        return w, minerror, bestdata
        
    def predict(self, dataMat, w):
        predictdata = self.sigmoid(dataMat * w)
        for i in range(len(predictdata)):
            if predictdata[i] > 0.5:#预测结果大于0.5，则分为1类
                predictdata[i] = 1
            else:
                predictdata[i] = 0#预测结果小于0.5，则分为0类
        return predictdata
#决策树类
class Stump():
    def __init__(self, numsteps = 10.0):
        self.numsteps = numsteps

    def stumpClassify(self,dataMatrix,dimen,threshVal,threshIneq):
        retArray = np.ones((np.shape(dataMatrix)[0],1))#创建一个初始值为1的大小为（m，1）的数组
        if(threshIneq=='lt'):#lt表示小于的意思
            retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
        else:
            retArray[dataMatrix[:,dimen]>threshVal]=-1.0
        return retArray

    def fit(self,dataArr, classLabels, D):
        dataMatrix=np.mat(dataArr)#转化为二维矩阵，而且只适应于二维
        labelMat = np.mat(classLabels).T#矩阵的转置
        m,n = np.shape(dataMatrix)
        bestStump={}#存储给定权重向量D时所得到的最佳单层决策树
        bestClasEst = np.mat(np.zeros((m,1)))
        minError = float('inf')#正无穷大
        for i in range(n):#第一层循环，进行两次循环，每次针对一个属性值
            #获取当前属性值的最大最小值
            rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max()
            stepSize = (rangeMax-rangeMin)/self.numsteps# 获取步长
            for j in range(-1,int(self.numsteps)+1):
            #for j in range(-1,int(numSteps)+1):#从-1开始，
                #目的是解决小值标签为-1 还是大值标签为-1的问题
                for inequal in ['lt','gt']:#为什么要有这么一个循环？
                    threshVal = (rangeMin+float(j)*stepSize)#从-1步开始，每走一步的值
                    predictedVals = self.stumpClassify(dataMatrix,i,threshVal,inequal)#根据这个值进行数据分类
                    errArr = np.mat(np.ones((m,1)))#创建错误向量，初始值为1
                    errArr[predictedVals==labelMat]=0#若预测的值和真实值相同，则赋值为0
                    weightedError = D.T*errArr#权重与错误向量相乘求和
                    if weightedError<minError:#此if语句在于获取最小的 错误权值
                        minError = weightedError
                        bestClasEst = predictedVals.copy()
                        bestStump['dim']=i
                        bestStump['thresh']=threshVal
                        bestStump['ineq'] = inequal
        return bestStump,minError,bestClasEst
#Adaboost类
class Adaboost():
    def __init__(self, base):
        '''
        :param base: 基分类器编号 0 代表对数几率回归 1 代表决策树桩
        在此函数中对模型相关参数进行初始化，如后续还需要使用参数请赋初始值
        '''
        self.base = base
        self.index = []
        self.classfierarray = []
        self.alpha = []
        self.classfierArray = {}
        
    #进行数据预处理函数，选择57个特征属性中相关性最高的30个
    def process(self, x_file, y_file):
        x,y = loadData(x_file, y_file)
        x = np.mat(x)
        y = np.array(y)
        x = autoNorm(x)
        p = []
        for i in range(x.shape[1]):
            pccs = pearson(x[:,i],y)
            p.append([pccs,i])
        t = sorted(p,key= lambda x:x[0])
        index = []
        for i in range(1,31):
            index.append(t[-i][1])
        self.index = index
        data = []
        for i in index:
            data.append(x[:,i])
        data = np.array(data).T
        data = data.reshape(data.shape[1],data.shape[2])
        return data, y

    def fit_(self,traindata,trainclasslabels,base_num):
        if(self.base == 0):
            weakClassArr = []
            alphas = []
            m = np.shape(trainclasslabels)[0]#获取数据个数
            D = np.mat(np.ones((m,1))/m)#初始化数据权值
            labels = trainclasslabels.copy()
            labels[np.where(labels == 0)] = -1
            for i in range(base_num):#循环
                estimator = logical()
                bestclass,error,classEst = estimator.stocGradAscent1(np.array(traindata),trainclasslabels,D)  
                #le-16 作用是保证没有除0溢出错误发生        
                classEst[np.where(classEst == 0)] = -1
                alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))#eN表示10的N次方
                alphas.append(alpha)
                weakClassArr.append(bestclass)
                expon = np.multiply(-1 * alpha * np.mat(labels).T,classEst)
                D = np.multiply(D,np.exp(expon))   
                D = D/D.sum()#用于计算下一次迭代中的新权重向量D
            return weakClassArr,alphas
        else:
            weakClass = []
            m = np.shape(traindata)[0]#获取数据个数
            D = np.mat(np.ones((m,1))/m)#初始化数据权值
            for i in range(base_num):#循环
                estimator = Stump()
                bestStump,error,classEst = estimator.fit(traindata,trainclasslabels,D)#寻找最优单层决策树
                #该值是说明总分类器单层决策树输出结果的权重？
                #le-16 作用是保证没有除0溢出错误发生
                alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))#eN表示10的N次方
                bestStump['alpha'] = alpha
                weakClass.append(bestStump)
                expon = np.multiply(-1*alpha*np.mat(trainclasslabels).T,classEst)
                D = np.multiply(D,np.exp(expon))
                D = D/D.sum()#用于计算下一次迭代中的新权重向量D
            return weakClass
    
    def train_and_predict(self, traindata, trainclasslabels, base_num):
        length = int(len(traindata)/10)
        for i in range(1,11):
            testdatas = traindata[(i-1)*length:i*length]
            if (i ==1):
                traindatas = traindata[length:len(traindata)]
                trainclasslabel = trainclasslabels[length:len(traindata)]
            elif (i==10):
                traindatas = traindata[:9*length]
                trainclasslabel = trainclasslabels[:9*length]
            else: 
                traindatas = np.append(traindata[:(i-1)*length],traindata[i * length:len(traindata)],axis=0)
                trainclasslabel = np.append(trainclasslabels[:(i-1)*length],trainclasslabels[i * length:len(traindata)],axis=0) 
            if(self.base == 0):
                print(i)
                classifierArray,alpha = self.fit_(traindatas,trainclasslabel,base_num)
                result = self.predict_data_0(classifierArray,alpha,testdatas)
                a = np.array(result).flatten().tolist()      
            else:
                classifierArray = self.fit_(traindatas,trainclasslabel,base_num)
                if(base_num == 100):
                    self.classfierArray = classifierArray
                result = self.predict_data_1(testdatas,classifierArray)
                result[np.where(result == -1)] = 0
                a = np.array(result).flatten().tolist()
            self.data_write_csv(base_num,i,a)

    def data_write_csv(self, base_num, num, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
        current_path = os.path.dirname(__file__)
        length = len(datas)
        base_list = [1,5,10,100]
        if(self.base == 1):
            f = open(current_path + '/experiments/base%d_fold%d.csv' % (base_num, num), "w")
        else :
            f = open(current_path + '/experiments/base%d_fold%d.csv' % (base_list[base_num - 1], num), "w")
        for j in range(0,length):
            f.write(str(j + 1 + (num - 1) * length)+","+str(datas[j])+"\n")
        f.close()

    def fit(self, x_file, y_file):
        '''
        在此函数中训练模型
        :param x_file:训练数据(data.csv)
        :param y_file:训练数据标记(targets.csv)
        '''
        if(self.base == 1):
            dataMat,trainclasslabels = loadData(x_file, y_file)
            traindata = np.mat(dataMat)
            trainclasslabels = np.array(trainclasslabels)
            trainclasslabels[np.where(trainclasslabels == 0)] = -1
            base_list = [1,5,10,100]
            for base_num in base_list:
                self.train_and_predict(traindata,trainclasslabels,base_num)
        else:
            dataset,trainclasslabels = self.process(x_file, y_file)
            datasets,trainclasslabel = loadData(x_file, y_file)
            b = np.ones((dataset.shape[0],1))
            dataset = np.append(dataset, b, axis=1)
            dataset = np.mat(dataset)
            trainclasslabels = np.array(trainclasslabels)
            datasets = np.append(datasets, b, axis=1)
            datasets = np.mat(datasets)
            trainclasslabel = np.mat(trainclasslabel).T
            test = Logistic()
            self.classfierarray = test.gradeDesc(datasets,trainclasslabel)
            self.alpha.append(1)
            base_list = [1,2,3,4]
            for base_num in base_list:
                self.train_and_predict(dataset,trainclasslabels,base_num)
        current_path = os.path.dirname(__file__)
        target = np.genfromtxt(current_path + '/targets.csv')
        base_list = [1, 5, 10, 100]
        for base_num in base_list:
            acc = []
            for i in range(1, 11):
                fold = np.genfromtxt(current_path + '/experiments/base%d_fold%d.csv' % (base_num, i), delimiter=',', dtype=np.int)
                accuracy = sum(target[fold[:, 0] - 1] == fold[:, 1]) / fold.shape[0]
                acc.append(accuracy)
            print(base_num,"个基下准确率：",np.array(acc).mean())

    def predict_data_0(self,cless,alpha,testdata):
        m = np.shape(testdata)[0]
        result = []
        aggClassEst = np.mat(np.zeros((m,1)))
        for i in range(len(cless)):
            estimator = logical()
            classEst = estimator.predict(testdata,np.mat(cless[i]).T)
            classEst[np.where(classEst == 0)] = -1
            aggClassEst += alpha[i] * classEst
        result = np.sign(aggClassEst)
        result[np.where(result == -1)] = 0
        return result

    def predict_data_1(self, testdata, classifierArray):
        m = np.shape(testdata)[0]
        aggClassEst = np.mat(np.zeros((m,1)))
        for i in range(len(classifierArray)):
            estimator = Stump()
            classEst = estimator.stumpClassify(testdata,classifierArray[i]['dim'],classifierArray[i]['thresh'],
                                        classifierArray[i]['ineq'])
            aggClassEst+=classifierArray[i]['alpha']*classEst
        result = np.sign(aggClassEst)
        return result

    def predict(self, x_file):
        '''
        :param x_file:测试集文件夹(后缀为csv)
        :return: 训练模型对测试集的预测标记
        '''
        testdata = loaddatas(x_file)
        testdata = np.mat(testdata)
        if(self.base == 0):
            b = np.ones((testdata.shape[0],1))
            testdata = np.append(testdata, b, axis=1)
            testdata = np.mat(testdata)
            test = Logistic()
            result = test.predict(testdata,self.classfierarray)
            #result = self.predict_data_0(self.classfierarray,self.alpha,testdata)
            a = np.array(result).flatten().tolist()
        else :
            result = self.predict_data_1(testdata,self.classfierArray)
            result[np.where(result == -1)] = 0
            a = np.array(result).flatten().tolist()
        return a




test1 = Adaboost(base = 0)
test1.fit("data.csv", "targets.csv")
data = test1.predict("data.csv")

