# NUMPY_NOTE #
[参考官方手册](https://docs.scipy.org/doc/numpy/user/quickstart.html)

## 基础 ##

numpy的对象为n维的数组ndarray。

ndarray.ndim--ndarray的维度

ndarray.shape--ndarray的具体形状 例如(3,2)代表一个三行两列的数组

ndarray.size--ndarray的元素数目

ndarray.dtype--ndarray中元素的数据类型

ndarray.itemsize--ndarray中元素的尺寸

ndarray.data



#### example ####

    >>> import numpy as np
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
          [ 5,  6,  7,  8,  9],
          [10, 11, 12, 13, 14]])
    >>> a.shape
    (3, 5)
    >>> a.ndim
    2
    >>> a.dtype.name
    'int64'
    >>> a.itemsize
    8
    >>> a.size
    15
    >>> type(a)
    <type 'numpy.ndarray'>
    >>> b = np.array([6, 7, 8])
    >>> b
    array([6, 7, 8])
    >>> type(b)
    <type 'numpy.ndarray'>


## 创建数组 ##

一. 用list创建一维数组
 
    >> import numpy as np
    >>> a = np.array([2,3,4])
    >>> a
    array([2, 3, 4])
    >>> a.dtype
    dtype('int64')
    >>> b = np.array([1.2, 3.5, 5.1])
    >>> b.dtype
    dtype('float64')
二. 创建多维数组

    >>> b = np.array([(1.5,2,3), (4,5,6)])
    >>> b
    array([[ 1.5,  2. ,  3. ],
           [ 4. ,  5. ,  6. ]])

三. 通常情况下，数组的元素是位置的，但是其尺寸是已知的，我们可以通过numpy的一些方法创建数组。

    >>> np.zeros( (3,4) )#创建3*4的全零数组
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.]])
    >>> np.ones( (2,3,4), dtype=np.int16 )#创建2*3*4的全一数组
    array([[[ 1, 1, 1, 1],
            [ 1, 1, 1, 1],
            [ 1, 1, 1, 1]],
           [[ 1, 1, 1, 1],
            [ 1, 1, 1, 1],
            [ 1, 1, 1, 1]]], dtype=int16)
    >>> np.empty( (2,3) ) #创建2*3的初值未知的数组
    array([[  3.73603959e-262,   6.02658058e-154,     6.55490914e-260],
           [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])

四. 使用arange或linspace创建一个序列

    >>> np.arange( 10, 30, 5 )
    array([10, 15, 20, 25])
    >>> np.arange( 0, 2, 0.3 ) # it accepts float arguments
    array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])


    >>> from numpy import pi
    >>> np.linspace( 0, 2, 9 ) # 9 numbers from 0 to 2
    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
    >>> x = np.linspace( 0, 2*pi, 100 )# useful to evaluate function at lots of points
    >>> f = np.sin(x)

## 基本运算 ##
    >>> a = np.array( [20,30,40,50] )
    >>> b = np.arange( 4 )
    >>> b
    array([0, 1, 2, 3])
    >>> c = a-b              #对应元素相减
    >>> c
    array([20, 29, 38, 47])
    >>> b**2 #每个元素分别平方
    array([0, 1, 4, 9])
    >>> 10*np.sin(a)  #每个元素求正弦
    array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])  
    >>> a<35     #元素小于35为真
    array([ True, True, False, False])

#### 元素相乘与矩阵乘法 ####

    >>> A = np.array( [[1,1], [0,1]] )
    >>> B = np.array( [[2,0],[3,4]] )
    >>> A * B        # elementwise product
    array([[2, 0],[0, 4]])
    >>> A @ B        # matrix product
    array([[5, 4], [3, 4]])
    >>> A.dot(B)     # another matrix product
    array([[5, 4],[3, 4]])

#### +=和*=操作 ####

    >>> a = np.ones((2,3), dtype=int)
    >>> b = np.random.random((2,3))
    >>> a *= 3 #a中每个元素都乘3
    >>> a
    array([[3, 3, 3],
       [3, 3, 3]])
    >>> b += a #对应相加
    >>> b
    array([[ 3.417022  ,  3.72032449,  3.00011437],
       [ 3.30233257,  3.14675589,  3.09233859]])
    >>> a += b  # b为浮点型 不会自动转换为整形 因此会报错
    Traceback (most recent call last):
      ...
    TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
#### 计算数组元素的和 最大值 最小值 ####

    >>> a = np.random.random((2,3))
    >>> a
    array([[ 0.18626021,  0.34556073,  0.39676747],
           [ 0.53881673,  0.41919451,  0.6852195 ]])
    >>> a.sum()
     2.5718191614547998
    >>> a.min()
     0.1862602113776709
    >>> a.max()
     0.6852195003967595

利用axis参数可以计算行或列的和 最值

    >>> b = np.arange(12).reshape(3,4)
    >>> b
    array([[ 0,  1,  2,  3],
          [ 4,  5,  6,  7],
          [ 8,  9, 10, 11]])
    >>>
    >>> b.sum(axis=0)# sum of each column
    array([12, 15, 18, 21])
    >>>
    >>> b.min(axis=1)# min of each row
    array([0, 4, 8])
    >>>
    >>> b.cumsum(axis=1) # cumulative sum along each row
    array([[ 0,  1,  3,  6],
          [ 4,  9, 15, 22],
          [ 8, 17, 27, 38]])

## 切片和遍历 ##
#### 切片略 ####
#### 遍历 ####
    >>> for row in b:
    ... print(row)  #按列遍历
    ...
    [0 1 2 3]
    [10 11 12 13]
    [20 21 22 23]
    [30 31 32 33]
    [40 41 42 43]


    >>> for element in b.flat:
    ... print(element) #按元素遍历
    ...
    0
    1
    2
    3
    10
    11
    12
    13
    20
    21
    22
    23
    30
    31
    32
    33
    40
    41
    42
    43

## 改变形状 ##

下面三种方法都可以改变形状 但都不会改变原来的数组

    >>> a.ravel()  # returns the array, flattened 铺平
    array([ 2.,  8.,  0.,  6.,  4.,  5.,  1.,  1.,  8.,  9.,  3.,  6.])




    >>> a.reshape(6,2)  # returns the array with a modified shape 返回修改过的形状
    array([[ 2.,  8.],
	       [ 0.,  6.],
	       [ 4.,  5.],
	       [ 1.,  1.],
	       [ 8.,  9.],
	       [ 3.,  6.]])
  


    >>> a.T  # returns the array, transposed 转置
    array([[ 2.,  4.,  8.],
	       [ 8.,  5.,  9.],
	       [ 0.,  1.,  3.],
	       [ 6.,  1.,  6.]])
    >>> a.T.shape
    (4, 3)
    >>> a.shape
    (3, 4)

resize()方法修改原数组的形状

    >>> a
    array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
    >>> a.resize((2,6))
    >>> a
    array([[ 2.,  8.,  0.,  6.,  4.,  5.],
          [ 1.,  1.,  8.,  9.,  3.,  6.]])

-1代表此维上数字自动计算

    >>> a.reshape(3,-1)
    array([[ 2.,  8.,  0.,  6.],
	       [ 4.,  5.,  1.,  1.],
	       [ 8.,  9.,  3.,  6.]])

## 合并数组 ##
## Copies and View ##
当操作一个数组时，数据有时候会复制到另一个数组有时候却不会，这经常是初学者困惑的来源。
#### No Copy at All ####
简单的赋值并不会复制数组的数据

    >>> a = np.arange(12)
    >>> b = a  # no new object is created
    >>> b is a   # a and b are two names for the same ndarray object
    True
    >>> b.shape = 3,4# changes the shape of a
    >>> a.shape
    (3, 4)

#### View or Shallow Copy ####
不同的数组元素可以共享相同的数据，view方法可以创造一个对象looks at the same data.

    >>> c = a.view()
    >>> c is a
    False
    >>> c.base is a# c is a view of the data owned by a
    True
    >>> c.flags.owndata
    False
    >>>
    >>> c.shape = 2,6  # a's shape doesn't change
    >>> a.shape
    (3, 4)
    >>> c[0,4] = 1234  # a's data changes
    >>> a
    array([[   0,1,2,3],
	       [1234,5,6,7],
	       [   8,9,   10,   11]])

对一个数组进行切片返回的是数组的view

    >>> s = a[ : , 1:3] # spaces added for clarity; could also be written "s = a[:,1:3]"
    >>> s[:] = 10   # s[:] is a view of s. Note the difference between s=10 and s[:]=10
    >>> a
    array([[   0,   10,   10,3],
	       [1234,   10,   10,7],
	       [   8,   10,   10,   11]])
#### Deep Copy ####
copy方法产生一个新的对象，复制原来的数据到新的内存中，和原数据互不相干
    >>> d = a.copy()  # a new array object with new data is created
    >>> d is a
    False
    >>> d.base is a   # d doesn't share anything with a
    False
    >>> d[0,0] = 9999
    >>> a
    array([[   0,   10,   10,3],
	       [1234,   10,   10,7],
	       [   8,   10,   10,   11]])

## 基本线性代数运算 Linear Algebar ##

    >>> import numpy as np
    >>> a = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> print(a)
    [[ 1.  2.]
     [ 3.  4.]]
    
    >>> a.transpose()
    array([[ 1.,  3.],
           [ 2.,  4.]])
     
    >>> np.linalg.inv(a)  #求逆矩阵
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])
    
    >>> u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
    >>> u
    array([[ 1.,  0.],
           [ 0.,  1.]])
    >>> j = np.array([[0.0, -1.0], [1.0, 0.0]])
    
    >>> j @ j# matrix product
    array([[-1.,  0.],
           [ 0., -1.]])
    
    >>> np.trace(u)  # trace
       2.0
    
    >>> y = np.array([[5.], [7.]])
    >>> np.linalg.solve(a, y) #解方程
    array([[-3.],
           [ 4.]])
    
    >>> np.linalg.eig(j) #计算特征向量
    (array([ 0.+1.j,  0.-1.j]), array([[ 0.70710678+0.j,  0.70710678-0.j],
       [ 0.00000000-0.70710678j,  0.00000000+0.70710678j]]))