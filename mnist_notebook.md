# MNIST 手写数字识别
## 数据的读入
用padas来读取csv格式的文件
    
    train = pd.read_csv("..\input\train.csv")
    
    test = pd.read_csv("..\input\test.csv")

读取的数据类型为padas的DataFrame类型

## 提取出label
训练集的第一列为label 其column为 label

     Y_train = train['label']

删掉label这一列
    
    X_train = train.drop(label = 'label', axis = 1)
    

删除train 释放空间

    del train

## 检查数据
    Y_train.value_count()
    X_train.isnull().any().describe()
    test.isnull().any.describe()

any()函数作用为若传入参数全为false 则输出为false 若有一个为true则输出为true  与full函数相反

describe()函数生成数据中元素的总数量（排除掉NAN）

value_count()函数统计相同的值的个数
以上均需要print函数才能输出

## 正则化
    X_train = X_train/255
    test = test/255

## reshape


    arr.ravel()  # 此函数为将arr拉平为一维数组

    X_train.value.reshape(-1, 28, 28, 1)
    test.value.reshape(-1, 28, 28, 1)
-1代表自动推算出正确维度  

    print(X_train.shape(), text.shape())#检查维度

    Y_train = Y_train.tocategorical(Y_train, num_classes=10)  #将标签向量化

## 分离出训练集和验证集
    random_seed = 2
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
从训练集中随机分成两部分 一部分作为训练 另一部分用来做验证集 验证集比例为0.1 

四个参数分别为训练集，训练集标签  验证集所占比例 随机种子

## 卷积神经网络建模 ##
    model = Squential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='Same', activation='relu',
                 input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='Same', activation='relu',))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='Same', activation='relu',))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='Same', activation='relu',))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='Same', activation='relu',))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='Same', activation='relu',))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
   
## 编译模型 ##
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    

## 进行训练 ##
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=0.00001)
    
    
    epochs = 5  # Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 86
    
    history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
      validation_data = (X_val, Y_val), verbose = 2)

## 在测试集进行测试 ##
    results = model.predict(test)

## 输出结果处理 ##
    results = np.argmax(results, axis=1)  # 在列上取最大值
    
    results = pd.Series(results, name="Label")  # 初始化为Series
    
    submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)  # 将两个Series拼成两列
    
    submission.to_csv("cnn_mnist_datagen.csv", index=False)
    