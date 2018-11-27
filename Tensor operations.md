# Tensor操作

*refer to: https://www.tensorflow.org/api_docs/python/tf*

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

