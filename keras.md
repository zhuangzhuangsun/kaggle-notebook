# keras总结 #
#### [参考keras中文文档](https://keras-cn.readthedocs.io/en/latest/) ####

## Sequential模型 ##
Squential模型为多个网络层线性堆叠

### 使用前先import ###

    from keras.model import Sequential

    from keras.layers import Dense, Activation


### 新建一个Squential对象 ###

    model = Sequential()
### 用.add()方法一个个的添加layer ###
 
    model.add(Dense(32, input_shape=(784,)))


    model.add(Activation('relu'))

Squential的第一层需要一个输入数据shape参数，后面各个层可以自动推导出中间数据的shape。为第一层制定输入数据的方法为传递一个Input_shape的关键字给第一层，它是一个tuple类型的数据。可以填入None，代表该位置可能为任意整数

    model.add(Conv2D(filters=32, input_shape=(28,28,1)))

## 编译 ##
在训练模型之前，需要通过compile来对学习过程进行配置。
compile接受三个参数

- optimizer 该参数可指定为预先定义的优化器名，也可自定义
- 损失函数loss：改参数为模型试图最小化的目标函数 可为预定义的损失函数名，如categorical——crossentropy、mse。也可以为一个自定义的损失函数。
- 指标列表metrics 对分类问题一般设置为metrics=['accuracy']。指标可以是一个预定义的名字，也可以是定制的函数。

example:    

    注释#For a multi-class classification problem
     model.compile(optimizer='rmsprop',
      loss='categorical_crossentropy',
      metrics=['accuracy'])
   
    注释#For a binary classification problem
    
    model.compile(optimizer='rmsprop',
      loss='binary_crossentropy',
      metrics=['accuracy'])
    
    注释#For a mean squared error regression problem
    
    model.compile(optimizer='rmsprop',
      loss='mse')
       
     
    注释#For custom metrics
    
    import keras.backend as K
    
    def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
    
    model.compile(optimizer='rmsprop',
      loss='binary_crossentropy',
      metrics=['accuracy', mean_pred])
    
## 训练 ##

keras使用numpyarray作为输入数据和标签的数据类型。训练模型一般用fit函数。

    fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

- x为输入数据，类型为numpy array或者是由numpy array组成的list
- y为label 类型为numpy array
- batch_size 为每个batch的大小 每训练一个batch 参数优化一次
- epochs为训练轮数
- verbose为日志显示 0：不显示 1：输出进度条记录 2：每个epoch输出一行记录
- callbacks:默认为None
- validation_spilt:0~1之间的小数，把训练集抽出一部分用于验证
- validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt
- shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱
- class_weight：
- sample_weight：
- initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用


fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况

## predict ##
    predict(self, x, batch_size=32, verbose=0)
batch_size不写的话为全部数据为一个batch

#### [更多详见序贯模型API](https://keras-cn.readthedocs.io/en/latest/models/sequential/) ####


## 图片预处理 ##

TO BE CONTINUE