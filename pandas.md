# pandas_notebook #
> 
#### [详见pandas documentations](http://pandas.pydata.org/pandas-docs/stable/10min.html#setting) ####
## import ##
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
## Objict Creation ##
Serise可以通过 list来初始化，label默认为0到n-1

    s = pd.Serises([1, 3, 5, 6, np.nan])
    print(s)

DataFrame可以通过一个numpy array初始化
       

    In [6]: dates = pd.date_range('20130101', periods=6)
    In [7]: dates
    Out[7]: 
    DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
       '2013-01-05', '2013-01-06'],
      dtype='datetime64[ns]', freq='D')
    
    In [8]: df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
    
    In [9]: df
    Out[9]: 
                     A         B        C         D
    2013-01-01  0.469112 -0.282863 -1.509059 -1.135632
    2013-01-02  1.212112 -0.173215  0.119209 -1.044236
    2013-01-03 -0.861849 -2.104569 -0.494929  1.071804
    2013-01-04  0.721555 -0.706771 -1.039575  0.271860
    2013-01-05 -0.424972  0.567020  0.276232 -1.087401
    2013-01-06 -0.673690  0.113648 -1.478427  0.524988

也可以通过一个dictionary来初始化

     In [10]: df2 = pd.DataFrame(
       ....:{ 'A' : 1.,
       ....:  'B' : pd.Timestamp('20130102'),
       ....:  'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
       ....:  'D' : np.array([3] * 4,dtype='int32'),
       ....:  'E' : pd.Categorical(["test","train","test","train"]),
       ....:  'F' : 'foo' }
                               )
    In [11]: df2
    Out[11]: 
         A      B       C   D    E     F
    0  1.0 2013-01-02  1.0  3   test  foo
    1  1.0 2013-01-02  1.0  3  train  foo
    2  1.0 2013-01-02  1.0  3   test  foo
    3  1.0 2013-01-02  1.0  3  train  foo
## 查看数据 ##
    df.head(3) # 查看第三行数据
    df.tial(3) # 查看最后三行数据
    df.index # 查看Index
    df.columns # 查看cloumn
    df.values #查看数据

    IN: df.describe() # 查看数据的统计值
    Out: 
                A        B         C         D
    count  6.000000  6.000000  6.000000  6.000000
    mean   0.073711 -0.431125 -0.687758 -0.233103
    std    0.843157  0.922818  0.779887  0.973118
    min   -0.861849 -2.104569 -1.509059 -1.135632
    25%   -0.611510 -0.600794 -1.368714 -1.076610
    50%    0.022070 -0.228039 -0.767252 -0.386188
    75%    0.658444  0.041933 -0.034326  0.461706
    max    1.212112  0.567020  0.276232  1.071804
    
    df.info() # 输出df每一列未缺失的数据个数和数据类型
    
    df.T # 转置
## 排序 ##

    df.sort_index(axis=1, ascending=False) # 按列名升序排列 
    df.sort_vlaue(by = 'B', # 按照某一行(列)排序，若axis为0(1),则填行名(列名)，数据类型为str or list of str
                  axis = 1 , # 指定按行(列)排序
                  ascending = # 布尔值 升序或者降序
                  inplace =   # 布尔值 是否用排序后的序列代替原序列
                  kind =      # str 排序方法 默认为 quicksort
                  na_position = # str 缺失值的方向 默认缺失值排在最后 )

## 切片 ##

    df['A'] # 切出column为A的这一列
    df[0:3] # 切出0到3列
    df['20130102':'20130104'] # 切出20130102 到20130104 这几行
    
    按标签切片
    df.loc['20130102':'20130104', ['A', 'B']]  切出行和列所交的部分 
    df.loc[data[0], ['A', 'B']]  # 切出A 和 B 列的第一个元素
    
    按元素位置item location切片
    df.iloc[3:5, 0:2] # 切出第四行到第五行，第一列到第二列的元素
    df.iloc[[1,2], [0,1]] # 切出第二行第三行 和 第一列第二列交的元素

## boolean indexing ##
    用一列的值选择数据
    In: df[df.A>0]
    out:
                        A        B          C         D
		2013-01-01  0.469112 -0.282863 -1.509059 -1.135632
		2013-01-02  1.212112 -0.173215  0.119209 -1.044236
		2013-01-04  0.721555 -0.706771 -1.039575  0.271860

    筛选出布尔值为正的值
    In: df[df>0]
    out: 
                       A         B         C         D
     2013-01-01  0.469112       NaN       NaN       NaN
     2013-01-02  1.212112       NaN  0.119209       NaN
     2013-01-03       NaN       NaN       NaN  1.071804
     2013-01-04  0.721555       NaN       NaN  0.271860
     2013-01-05       NaN  0.567020  0.276232       NaN
     2013-01-06       NaN  0.113648       NaN  0.524988

## 用 isin() 来挑选元素 ##
    In: df2 = df.copy()
    In: df2['E'] = ['one', 'one','two','three','four','three']
    In: df2
    Out: 
                    A          B        C         D       E
    2013-01-01  0.469112 -0.282863 -1.509059 -1.135632   one
    2013-01-02  1.212112 -0.173215  0.119209 -1.044236   one
    2013-01-03 -0.861849 -2.104569 -0.494929  1.071804   two
    2013-01-04  0.721555 -0.706771 -1.039575  0.271860  three
    2013-01-05 -0.424972  0.567020  0.276232 -1.087401   four
    2013-01-06 -0.673690  0.113648 -1.478427  0.524988  three
    
    In: df2[df2['E'].isin(['two','four'])]
    Out: 
                    A         B         C         D       E
    2013-01-03 -0.861849 -2.104569 -0.494929  1.071804   two
    2013-01-05 -0.424972  0.567020  0.276232 -1.087401  four

## 赋值 ##
    In: s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))

    In: s1
    Out: 
		    2013-01-02 1
		    2013-01-03 2
		    2013-01-04 3
		    2013-01-05 4
		    2013-01-06 5
		    2013-01-07 6
	        Freq: D, dtype: int64
    
    In: df['F'] = s1

使用标签进行赋值 at(assignment to)
  
    df.at[dates[0],'A'] = 0

利用元素位置进行赋值

    df.iat[0,1] = 0

使用numpyarray进行赋值

    df.loc[:,'D'] = np.array([5] * len(df))

## 对缺失的数据进行处理 np.nan ##

Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of the data.

    In: df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
    
    In: df1.loc[dates[0]:dates[1],'E'] = 1
    
    In: df1
    Out: 
                    A         B        C      D   F    E
    2013-01-01  0.000000  0.000000 -1.509059  5  NaN  1.0
    2013-01-02  1.212112 -0.173215  0.119209  5  1.0  1.0
    2013-01-03 -0.861849 -2.104569 -0.494929  5  2.0  NaN
    2013-01-04  0.721555 -0.706771 -1.039575  5  3.0  NaN

删掉有NAN的行(row)

    in: df1.dropna(how= 'any')
    Out: 
                   A         B         C      D   F    E

    2013-01-02  1.212112 -0.173215  0.119209  5  1.0  1.0

填充缺失的数据

    in: df1.fillna(value = 5)
    Out: 
                    A         B         C     D   F    E
    2013-01-01  0.000000  0.000000 -1.509059  5  5.0  1.0
    2013-01-02  1.212112 -0.173215  0.119209  5  1.0  1.0
    2013-01-03 -0.861849 -2.104569 -0.494929  5  2.0  5.0
    2013-01-04  0.721555 -0.706771 -1.039575  5  3.0  5.0

得到哪一个元素是缺失的布尔值

    In : pd.isna(df1)
    Out: 
                A      B      C      D      F      E
    2013-01-01  False  False  False  False   True  False
    2013-01-02  False  False  False  False  False  False
    2013-01-03  False  False  False  False  False   True
    2013-01-04  False  False  False  False  False   True


## 操作 ##
    
    df.mean() # 对列求均值
    df.mean(1) #对行求均值

不同维度的对象有时候需要运算 pandas自动根据特定的维度broadcast

    In: s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
    
    In: s
    Out: 
		    2013-01-01 NaN
		    2013-01-02 NaN
		    2013-01-03 1.0
		    2013-01-04 3.0
		    2013-01-05 5.0
		    2013-01-06 NaN
    Freq: D, dtype: float64
    In: df.sub(s, axis='index')
    Out: 
                   A         B         C    D    F
    2013-01-01   NaN        NaN       NaN   NaN  NaN
    2013-01-02   NaN        NaN       NaN   NaN  NaN
    2013-01-03 -1.861849 -3.104569 -1.494929  4.0  1.0
    2013-01-04 -2.278445 -3.706771 -4.039575  2.0  0.0
    2013-01-05 -5.424972 -4.432980 -4.723768  0.0 -1.0
    2013-01-06   NaN        NaN       NaN   NaN  NaN


Apply()方法：对对象使用括号内的运算

    df.apply(lambda x: x.max() - x.min()) # 每一列最大值减最小值
直方图化(Histogramming)

    s.value_count() # 对Series的元素数目进行统计得出元素的种类和值
字符串方法

    s.str.lower() # 将s中的字符串全变为小写


## Merge ##

    df.concat() # 参数为一个DataFrame的List，该函数将其按行拼接
    df.merge() # 该函数将DataFrame按列拼接
    df.append() # 该函数为DataFrame添加行或列

## Grouping ##
    In: df
    Out: 
	        A      B         C         D
	    0  foo    one   -1.202872 -0.055224
	    1  bar    one   -1.814470  2.395985
	    2  foo    two    1.018601  1.552825
	    3  bar   three   -0.595447  0.166599
	    4  foo    two    1.395433  0.047609
	    5  bar    two   -0.392670 -0.136473
	    6  foo    one    0.007207 -0.561757
	    7  foo   three   1.928123 -1.623033

    In : df.groupby('A').sum()
    Out: 
		            C        D
		    A 
		    bar -2.802588  2.42611
		    foo  3.146492 -0.63958
> 

    In : df.groupby(['A','B']).sum()
    Out: 
                    C        D
	    A   B
	   bar  one   -1.814470  2.395985`
	       three  -0.595447  0.166599
	        two   -0.392670 -0.136473
	   foo  one   -1.195665 -0.616981
	       three   1.928123 -1.623033
	        two    2.414034  1.600434

## Plotting ##
Series 作图

    ts.plot() # 横轴为index 纵轴为数据
    df.plot() #多条线 横轴为Index 纵轴为数据 每条线名称为column

## 数据in/out ##

csv文件(读出为DataFrame数据)
    
    df.to_csv('foo.csv')
    pd.read_csv('foo.csv')
    
