

# Tensor操作

*Refer to: https://www.tensorflow.org/api_docs/python/tf*

*TensorFlow ver: r1.12*



[TOC]

## 生成Tensor

### tf.constant()

API: https://www.tensorflow.org/api_docs/python/tf/constant

生成一个常量构成的tensor，常见的用法：

```python
>>> x1 = tf.constant([1, 2, 3, 4])
>>> x2 = tf.constant([1, 2, 3, 4, 5, 6], shape=(2, 3))
>>> x3 = tf.constant([1., 2., 3., 4.], shape=(2, 2))
>>> x2_1 = tf.constant([1, 2, 3, 4, 5, 6], shape=(2, 3), dtype='float16')
>>> x1
<tf.Tensor 'Const:0' shape=(4,) dtype=int32>
>>> x2
<tf.Tensor 'Const_1:0' shape=(2, 3) dtype=int32>
>>> x3
<tf.Tensor 'Const_2:0' shape=(2, 2) dtype=float32>
>>> x2_1
<tf.Tensor 'Const_3:0' shape=(2, 3) dtype=float16>
```

更加复杂的tensor可以先生成一个numpy.ndarray, 再将这个ndarray转换成tensor



### tf.range()

API: https://www.tensorflow.org/api_docs/python/tf/range

```python
tf.range(start, limit, delta)
```

**注意与tf.keras.backend.arange的区分：**

API: https://www.tensorflow.org/api_docs/python/tf/keras/backend/arange

```python
tf.keras.backend.arange(
    start, stop=None,
    step=1,
    dtype='int32'
)
```



### tf.placeholder()

API: https://www.tensorflow.org/api_docs/python/tf/placeholder

生成一个占位的tensor，一般用作模型的输入，比如输入的特征和标签

```python
tf.placeholder(
    dtype,
    shape=None,
    name=None
)
```

在tf.Session().run(xxx) 的时候，需要用`feed_dict`来填充这个占位的tensor：

```python
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
```



## Tensor形状有关操作

和tensor形状有关的操作



### tf.shape()

API: https://www.tensorflow.org/api_docs/python/tf/shape

获取tensor的shape，注意返回的是一个tensor，*~~虽然你知道他的形状，但是他仍然是个tensor~~*

```
>>> t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
>>> tf.shape(t)  # [2, 2, 3]
<tf.Tensor 'Shape_3:0' shape=(3,) dtype=int32>
```



### tf.reshape()

API: https://www.tensorflow.org/api_docs/python/tf/reshape

跟np.reshape()类似，改变一个tensor的形状

```python
>>> x = tf.range(1, 11, 1)
>>> tf.Session().run(x)
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
>>> x = tf.reshape(x, [2, 5])
>>> x
<tf.Tensor 'Reshape:0' shape=(2, 5) dtype=int32>
```

对于形状确定的tensor, 在指定reshape之后的形状的时候，一个维度可以填-1,  会自动算出来这个维度：

```python
>>> x = tf.range(1, 33, 1)
>>> x = tf.reshape(x, (-1, 2, 2))
>>> x
<tf.Tensor 'Reshape_2:0' shape=(8, 2, 2) dtype=int32>

```



### tf.expand_dims()

API: https://www.tensorflow.org/api_docs/python/tf/expand_dims

扩展维度，可以形象理解成给指定的维度加中括号，可以参见上面讲解维度的例子，参数`axis`为几，就是给对应的维度加上一个括号，如为0的时候就是给tensor自己加一个括号，为1的时候就是第一维，为2的时候就是给第二维（也就是第一维的每个切片中的每个元素），这样操作的结果就是给tensor增加了一个维度

**`axis` 为-1时，指的是最里面的维度**

```python
>>> t = tf.constant([1,2])
>>> tf.expand_dims(t, 0)
<tf.Tensor 'ExpandDims_2:0' shape=(1, 2) dtype=int32>
>>> tf.expand_dims(t, 1)
<tf.Tensor 'ExpandDims_3:0' shape=(2, 1) dtype=int32>
>>> tf.expand_dims(t, -1)
<tf.Tensor 'ExpandDims_4:0' shape=(2, 1) dtype=int32>

```

```python
>>> t2 = tf.reshape(tf.range(1, 31, 1), (2, 3, 5))
>>> t2
<tf.Tensor 'Reshape_1:0' shape=(2, 3, 5) dtype=int32>
>>> tf.expand_dims(t2, 0)
<tf.Tensor 'ExpandDims_5:0' shape=(1, 2, 3, 5) dtype=int32>
>>> tf.expand_dims(t2, 2)
<tf.Tensor 'ExpandDims_6:0' shape=(2, 3, 1, 5) dtype=int32>
>>> tf.expand_dims(t2, 3)
<tf.Tensor 'ExpandDims_7:0' shape=(2, 3, 5, 1) dtype=int32>
>>> tf.expand_dims(t2, -1)
<tf.Tensor 'ExpandDims_8:0' shape=(2, 3, 5, 1) dtype=int32>

```



## 两个或多个tensor的操作

### tf.concat()

API: https://www.tensorflow.org/api_docs/python/tf/concat

两个tensor的拼接

```python
tf.concat(
    values,
    axis,
    name='concat'
)
```

第一个参数是需要拼接的两个tensor, 第二个参数是拼接的维度

```python
>>> t1 = tf.constant([[1, 2, 3], [4, 5, 6]])
>>> t2 = tf.constant([[11, 22, 33], [44, 55, 66]])
>>> tf.concat([t1, t2], 0)
<tf.Tensor 'concat:0' shape=(4, 3) dtype=int32>
>>> tf.concat([t1, t2], 1)
<tf.Tensor 'concat_1:0' shape=(2, 6) dtype=int32>
>>> tf.concat([t1, t2], -1)
<tf.Tensor 'concat_2:0' shape=(2, 6) dtype=int32>
```

同样，`axis` 为-1时， 拼接最里面的维度

```python
>>> t1 = tf.constant([[[1, 2], [2, 3]], [[4, 4], [5, 3]]])
>>> t2 = tf.constant([[[7, 4], [8, 4]], [[2, 10], [15, 11]]])
>>> tf.concat([t1, t2], -1)
<tf.Tensor 'concat_3:0' shape=(2, 2, 4) dtype=int32>
```

拼接的两个tensor的形状不一样，但是需要保证能拼上

```>>> t3 = tf.constant([[[11, 22], [33, 44]]])
>>> t1 = tf.constant([[[1, 2], [2, 3]], [[4, 4], [5, 3]]])
>>> t3 = tf.constant([[[11, 22], [33, 44]]])
>>> tf.concat([t1, t3], 0)
<tf.Tensor 'concat_5:0' shape=(3, 2, 2) dtype=int32>
```

**注意与tf.keras.backend.concatenate的区别**

API: https://www.tensorflow.org/api_docs/python/tf/keras/backend/concatenate

```python
tf.keras.backend.concatenate(
    tensors,
    axis=-1
)
```

## Tensor的向量和矩阵运算

线性代数中常用到的一些运算，在TensorFlow中有一个专门的模块tf.linalg来进行线性代数的运算，这部分都会讲到。对应的，有一些数学运算可能会跟线性代数运算混淆，这里会单独列举



### tf.matmul()和tf.multiply()

矩阵乘法和矩阵元素的乘法

[矩阵乘法](https://zh.wikipedia.org/wiki/%E7%9F%A9%E9%99%A3%E4%B9%98%E6%B3%95)：可以看tf.matmul中的例子

矩阵元素相乘：要求两个矩阵的shape是一样的，计算结果是一个形状一样的矩阵，每个元素是相乘矩阵的对应位置的元素相乘

#### tf.matmul()

API: https://www.tensorflow.org/api_docs/python/tf/linalg/matmul

更准确一点应该是tf.linalg.matmul(), 但是写tf.matmul()也没问题，输出两个矩阵相乘的结果。我们直接看例子：
$$
\left[
 \begin{matrix}
   1 & 2 & 3 \\
   4 & 5 & 6
  \end{matrix}
  \tag{A}
  \right]
$$

$$
\left[
 \begin{matrix}
 7 & 8 \\
 9 & 10 \\
 11 & 12
 \end{matrix}
 \right] \tag{B}
$$

我们先来复习一下线性代数：

矩阵乘法结果是：
$$
A \times B = 
\left[
\begin{matrix}
\sum_{j=1}^{n} a_{1,j}*b_{j,1} & \sum_{j=1}^{n} a_{1,j}*b_{j,2} \\
\\
\sum_{j=1}^{n} a_{2,j}*b_{j,1} & \sum_{j=1}^{n} a_{2,j}*b_{j,2} \\
\end{matrix}
\right]
$$
然后看代码：

```python
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
c = tf.matmul(a,b)


print(c)
# Output: 
<tf.Tensor 'MatMul:0' shape=(2, 2) dtype=int32>

print(tf.Session().run(c))
# Output: 
array([[ 58,  64],
       [139, 154]])

```

一个更复杂一点的：
$$
\left[
\begin{matrix}


\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}



\begin{bmatrix}
7 & 8 & 9 \\
10 & 11 & 12
\end{bmatrix}


\end{matrix}
\right] \tag{A}
$$

$$
\left[
\begin{matrix}


\begin{bmatrix}
13 & 14 \\
15 & 16\\
17 & 18
\end{bmatrix}



\begin{bmatrix}
19 & 20\\
21 & 22\\
23 & 24
\end{bmatrix}


\end{matrix}
\right] 
\tag{B}
$$

```python
a = tf.constant(np.arange(1, 13, dtype=np.int32),
                shape=[2, 2, 3])
b = tf.constant(np.arange(13, 25, dtype=np.int32),
                shape=[2, 3, 2])
c = tf.matmul(a, b)

print(c)
# Output: 
Tensor("MatMul:0", shape=(2, 2, 2), dtype=int32)

print(tf.Session().run(c))
# Output: 
[[[ 94 100]
  [229 244]]

 [[508 532]
  [697 730]]]

```

输出的结果（继续复习线性代数）：
$$
\left[
\begin{matrix}


\begin{bmatrix}
94 & 100\\
229 & 244
\end{bmatrix}


\left[
\begin{matrix}
508& 532\\
697 & 730
\end{matrix}
\right]

\end{matrix}
\right]
$$
其实计算方式是：
$$
A = 
\left[
\begin{matrix}
M_{A-left} & M_{A-right}
\end{matrix}
\right]
\\
B = 
\left[
\begin{matrix}
M_{B-left} & M_{B-right}
\end{matrix}
\right]
\\
A \times B = 
\left[
\begin{matrix}
M_{A-left} * M_{B-left} & M_{A-right} * M_{B-right}
\end{matrix}
\right]
$$

#### tf.multiply()

API: https://www.tensorflow.org/api_docs/python/tf/math/multiply

```python
tf.math.multiply(
    x,
    y,
    name=None
)
```

- x和y数据类型需要相同，指的是dtype
- 也可以写成 x * y

公式：
$$
A = \left[
    \begin{matrix}
    a_{i,j}
    \end{matrix}
    \right]
\\
B = \left[
    \begin{matrix}
    b_{i,j}
    \end{matrix}
    \right]
\\

A \cdot B = 
\left[
    \begin{matrix}
    a_{i,j} \times b_{i,j}
    \end{matrix}
    
\right]
$$
代码：

```python
a = tf.constant(np.arange(7, 13, dtype=np.int32),
                shape=[2, 3])
b = tf.constant(np.arange(19, 25, dtype=np.int32),
                shape=[2, 3])

c = tf.multiply(a, b)

print(tf.Session().run(a))
# Output: 
[[ 7  8  9]
 [10 11 12]]

print(tf.Session().run(b))
# Output: 
[[19 20 21]
 [22 23 24]]

print(tf.Session().run(c))
# Output: 
[[133 160 189]
 [220 253 288]]
```

### tf.tensordot()

复习一下线性代数：

1. [张量积](https://zh.wikipedia.org/wiki/%E5%BC%A0%E9%87%8F%E7%A7%AF)
   $$
   \textbf b \otimes \textbf a = 
   \left[
       \begin{matrix}
       b_1\\
       b_2\\
       b_3\\
       b4
       \end{matrix}
       \right
   ]
   \left[
        \begin{matrix}
       a_1 & a_2 & a_3
       \end{matrix}
       \right
   ]
   =
   \left[
       \begin{matrix}
       a_1b_1 & a_2b_1 & a_3b_1\\
       a_1b_2 & a_2b_2 & a_3b_2\\
       a_1b_3 & a_2b_3 & a_3b_3\\
       a_1b_4 & a_2b_4 & a_3b_4
       \end{matrix}
       \right
   ]
   $$

2. [外积](https://zh.wikipedia.org/wiki/%E5%A4%96%E7%A7%AF)

   外积一般指两个向量的张量积

3. [克罗内克积](https://zh.wikipedia.org/wiki/%E5%85%8B%E7%BD%97%E5%86%85%E5%85%8B%E7%A7%AF)

   克罗内克积是[外积](https://zh.wikipedia.org/wiki/%E5%A4%96%E7%A7%AF)从向量到矩阵的推广
   $$
   A_{m \times n} = \begin{bmatrix}
   a_{i,j}
   \end{bmatrix}\\
   A \otimes B = 
   \begin{bmatrix}
   a_{1,1}B & \cdots & a_{1,n}B\\
   \vdots & \ddots & \vdots\\
   a_{m,1}B & \cdots & a_{m,n}B
   \end{bmatrix}
   $$
   结果是一个[分块矩阵](https://zh.wikipedia.org/wiki/%E5%88%86%E5%A1%8A%E7%9F%A9%E9%99%A3)


```python
tf.tensordot(
    a,
    b,
    axes,
    name=None
)
```

tensordot用起来比较麻烦，我们直接看下面几个例子：

- 对于一维的向量：


$$
\vec A = \left[
    \begin{matrix}
    a_{1}, & a_{2}, & \cdots & a_{m}
    \end{matrix}
\right]
\\
\vec B = \left[
    \begin{matrix}
    b_{1}, & b_{2}, & \cdots & b_{n}
    \end{matrix}
\right]
$$



```python
a = tf.constant(np.arange(7, 13, dtype=np.int32))
b = tf.constant(np.arange(19, 25, dtype=np.int32))
```

axes=1, 运算结果是向量点积，所以要求长度相同，即m = n
$$
\vec A \cdot \vec B = \sum_{i}^n a_i b_i
$$

```python
c = tf.tensordot(a, b, axes=1)
print(tf.Session().run(c))
# Output: 
1243
```

​	

	axes=0，运算结果是外积，m和n可以不相等, 运算结果是向量A和向量B的转置的外积：

$$
\textbf B^T = \left[
    \begin{matrix}
    b_1\\
    b_2\\
    \cdot\\
    b_m
    \end{matrix}
    \right]
\\
\quad \\
\textbf A \otimes \textbf{B}^T = [c_{i,j}]
\\
c_{i,j} = a_ib_j
$$

```python
c = tf.tensordot(a, b, axes=0)
print(tf.Session().run(c))
# Output:
[[133 140 147 154 161 168]
 [152 160 168 176 184 192]
 [171 180 189 198 207 216]
 [190 200 210 220 230 240]
 [209 220 231 242 253 264]
 [228 240 252 264 276 288]]
 
```

	注意这里a和b有先后关系，交换之后，计算的就是

$$
\textbf B \otimes \textbf{A}^T
$$

```python
c = tf.tensordot(b, a, axes=0)
print(tf.Session().run(c))
# Output:
[[133 152 171 190 209 228]
 [140 160 180 200 220 240]
 [147 168 189 210 231 252]
 [154 176 198 220 242 264]
 [161 184 207 230 253 276]
 [168 192 216 240 264 288]]
```

​	

- 二维的情况

  - axis=1时，计算矩阵乘法，因此a和b需要满足矩阵乘法的形状要求：
    $$
    A_{m \times n} = [a_{i,j}]=
    \left[
    \begin{matrix}
    \textbf A_1\\
    \textbf A_2\\
    \cdots\\
    \textbf A_m
    \end{matrix}
    \right]\\
    \quad\\
    B_{n \times m} = [b_{k,l}]=
    \left[
    \begin{matrix}
    \textbf B_1 & \textbf B_2 & \cdots & \textbf B_m
    \end{matrix}
    \right]\\
    \quad\\
    C_{m,m} = A \times B = [c_{o,p}]\\
    c_{o,p} = \textbf A_o\textbf B_p
    $$
    代码示例：

    ```python
    a = tf.constant([1,1,1,2,2,2], dtype=np.int32, shape=[2,3])
    b = tf.constant(np.arange(1, 7, dtype=np.int32), shape=[3,2])
    
    
    c1 = tf.tensordot(a, b, axes=1)
    c2 = tf.tensordot(b, a, axes=1)
    
    print(tf.Session().run(a))
    print(tf.Session().run(b))
    print(tf.Session().run(c1))
    print(tf.Session().run(c2))
    # Output: 
    
    [[1 1 1]
     [2 2 2]]
     
    [[1 2]
     [3 4]
     [5 6]]
     
    [[ 9 12]
     [18 24]]
     
    [[ 5  5  5]
     [11 11 11]
     [17 17 17]]
    ```


  - axis=0时,计算的是矩阵a中的每个元素数乘矩阵b，也就是克罗内克积：
    $$
    A_{m \times n} = [a_{i,j}]\\
    B = [b_{k,l}]\\
    result = C_{m \times n} = A \otimes B = 
    \begin{bmatrix}
    a_{i,j}B
    \end{bmatrix}
    $$
    **这里就体现出来了tensor和vector, matrix之间的区别来了**：

    在一维计算的时候，tensor和vector可以认为是等价的，tensor计算的结果可以是一个二维的tensor， vector的计算结果可能是一个矩阵，为了表示一个矩阵，我们可以用numpy.ndarray来表示：

    矩阵：
    $$
    \begin{bmatrix}133 & 152 & 171 & 190 & 209 & 228\\140 & 160 & 180 & 200 & 220 & 240\\147 & 168 & 189 & 210 & 231 & 252\\154 & 176 & 198 & 220 & 242 & 264\\161 & 184 & 207 & 230 & 253 & 276\\168 & 192 & 216 & 240 & 264 & 288\end{bmatrix}
    $$

```python
# print(arr)
# Output of
[[133 152 171 190 209 228]
 [140 160 180 200 220 240]
 [147 168 189 210 231 252]
 [154 176 198 220 242 264]
 [161 184 207 230 253 276]
 [168 192 216 240 264 288]]
```



目前为止还没什么问题，但是当运算的tensor是二维的时候，结果是一个分块矩阵，我们只能用
$$
A \otimes B = \begin{bmatrix}a_{1,1}B & \cdots & a_{1,n}B\\\vdots & \ddots & \vdots\\a_{m,1}B & \cdots & a_{m,n}B\end{bmatrix}
$$
这样的形式来表示这个矩阵，而不能是：
$$
A\otimes B={\begin{bmatrix}a_{11}b_{11}&a_{11}b_{12}&\cdots &a_{11}b_{1q}&\cdots &\cdots &a_{1n}b_{11}&a_{1n}b_{12}&\cdots &a_{1n}b_{1q}\\a_{11}b_{21}&a_{11}b_{22}&\cdots &a_{11}b_{2q}&\cdots &\cdots &a_{1n}b_{21}&a_{1n}b_{22}&\cdots &a_{1n}b_{2q}\\\vdots &\vdots &\ddots &\vdots &&&\vdots &\vdots &\ddots &\vdots \\a_{11}b_{p1}&a_{11}b_{p2}&\cdots &a_{11}b_{pq}&\cdots &\cdots &a_{1n}b_{p1}&a_{1n}b_{p2}&\cdots &a_{1n}b_{pq}\\\vdots &\vdots &&\vdots &\ddots &&\vdots &\vdots &&\vdots \\\vdots &\vdots &&\vdots &&\ddots &\vdots &\vdots &&\vdots \\a_{m1}b_{11}&a_{m1}b_{12}&\cdots &a_{m1}b_{1q}&\cdots &\cdots &a_{mn}b_{11}&a_{mn}b_{12}&\cdots &a_{mn}b_{1q}\\a_{m1}b_{21}&a_{m1}b_{22}&\cdots &a_{m1}b_{2q}&\cdots &\cdots &a_{mn}b_{21}&a_{mn}b_{22}&\cdots &a_{mn}b_{2q}\\\vdots &\vdots &\ddots &\vdots &&&\vdots &\vdots &\ddots &\vdots \\a_{m1}b_{p1}&a_{m1}b_{p2}&\cdots &a_{m1}b_{pq}&\cdots &\cdots &a_{mn}b_{p1}&a_{mn}b_{p2}&\cdots &a_{mn}b_{pq}\end{bmatrix}}
$$



​    我们打印一下形状：

```python
import tensorflow as tf
import numpy as np

a = tf.constant([1,1,1,2,2,2], dtype=np.int32, shape=[2,3])
b = tf.constant(np.arange(1, 7, dtype=np.int32), shape=[3,2])

c = tf.tensordot(a, b, axes=0)

print(tf.Session().run(c).shape)
# output: 
(2, 3, 3, 2)
```

上面的代码表明，ndarray/tensor中包含了分块矩阵的信息！

从形状我们可以看出来，这是一个两行三列的分块矩阵，其中的每个矩阵块又是一个三行两列的矩阵，也就是说ndarray/tensor除了储存了矩阵中元素的位置和数值信息，还储存了这个矩阵的分块信息。

当然，这个还只是tensor的维度是四维的情况下，再高维度tensor对应的就是分块矩阵中的每个矩阵块还是

