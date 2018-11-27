# TensorFlow basic

一些TensorFlow的基础知识和操作

## 什么是Tensor

1. tensor

   tensor在TensorFlow内部是对应矢量和矩阵的更高维度的数组，是TensorFlow内部的基本数据类型，表现的形式是n维的数组，例如一维的tensor: 

   ```python
   > > > import tensorflow as tf
   > > > import numpy as np
   > > >
   > > > array1 = np.array([1,2,3,4,5])
   > > > tensor1 = tf.constant(array1)
   > > > with tf.Session() as sess:
   > > > ...     print(tensor1)
   > > > ...     tensor1_value = sess.run(tensor1)
   > > > ...     print(tensor1_value)
   
   # Output:
   
   Tensor("Const:0", shape=(5,), dtype=int32)
   [1 2 3 4 5]
   ```



   可以看出来一个Tensor的实例包含的基本信息有`name`, `shape` 和 `dtype`. 而要知道一个tensor的数值，则需要通过tf.Session()来运行得到。


2. tensor的基本信息

   - `name`  是这个tensor在TensorFlow的图中的名字，可以看成是这个tensor的名字

   - `shape` 是这个tensor的形状，是一个tuple, tuple中的值从左到右依次是这个tensor从外到内的维度，可以参考下面的例子: 

     ```python
     >>> import tensorflow as tf
     >>> import numpy as np
     >>>
     >>> array1 = np.array([1,2,3,4,5])
     >>>
     >>> tensor1 = tf.constant(array1)
     >>> with tf.Session() as sess:
     ...     print(tensor1)
     ...     tensor1_value = sess.run(tensor1)
     ...     print(tensor1_value)
     ...     tensor2 = tf.expand_dims(tensor1, axis=-1)
     ...     print(tensor2)
     ...     tensor2_value = sess.run(tensor2)
     ...     print(tensor2_value)
     ...
     # Output: 
     Tensor("Const_1:0", shape=(5,), dtype=int32)
     [1 2 3 4 5]
     Tensor("ExpandDims:0", shape=(5, 1), dtype=int32)
     [[1]
      [2]
      [3]
      [4]
      [5]]
     ```

   - `dtype`  tensor中的数据的类型，对应的是numpy中的数据类型

   - tensor的**维度( dimension)**: 对tensor做修改的时候，经常需要一个参数是axis，这里就涉及到tensor的维度，我们可以从下面`expand_dims`  这个函数来看一下tensor的维度：

     ```python
     >>> import tensorflow as tf
     >>> import numpy as np
     >>>
     >>> array1 = np.array([[1,2,3,4,5],[2,3,4,5,6]])
     >>> tensor1 = tf.constant(array1)
     >>> with tf.Session() as sess:
     ...     print(tensor1)
     ...     tensor1_value = sess.run(tensor1)
     ...     print(tensor1_value)
     ...     tensor2 = tf.expand_dims(tensor1, axis=0)
     ...     print(tensor2)
     ...     tensor2_value = sess.run(tensor2)
     ...     print(tensor2_value)
     ...     tensor3 = tf.expand_dims(tensor1, axis=1)
     ...     print(tensor3)
     ...     tensor3_value = sess.run(tensor3)
     ...     print(tensor3_value)
     ...     tensor4 = tf.expand_dims(tensor1, axis=2)
     ...     print(tensor4)
     ...     tensor4_value = sess.run(tensor4)
     ...     print(tensor4_value)
     ...     tensor5 = tf.expand_dims(tensor1, axis=-1)
     ...     print(tensor5)
     ...     tensor5_value = sess.run(tensor5)
     ...     print(tensor5_value)
     
     # Output: 
     Tensor("Const:0", shape=(2, 5), dtype=int32)
     [[1 2 3 4 5]
      [2 3 4 5 6]]
     Tensor("ExpandDims:0", shape=(1, 2, 5), dtype=int32)
     [[[1 2 3 4 5]
       [2 3 4 5 6]]]
     Tensor("ExpandDims_1:0", shape=(2, 1, 5), dtype=int32)
     [[[1 2 3 4 5]]
     
      [[2 3 4 5 6]]]
     Tensor("ExpandDims_2:0", shape=(2, 5, 1), dtype=int32)
     [[[1]
       [2]
       [3]
       [4]
       [5]]
     
      [[2]
       [3]
       [4]
       [5]
       [6]]]
     Tensor("ExpandDims_3:0", shape=(2, 5, 1), dtype=int32)
     [[[1]
       [2]
       [3]
       [4]
       [5]]
     
      [[2]
       [3]
       [4]
       [5]
       [6]]]
     
     ```

     从上面的代码可以看出来，0, 1, 2, -1分别代表了这个tensor的最外面的维度，第一个维度，第二个维度和最里面的维度。要注意看一下axis=0和axis=1的区别。

3. tensor和numpy.ndarray之间的关系

   - tensor和ndarray之间可以相互转换，最常见的，可以通过一个ndarray来生成一个tensor，而一个tensor在TensorFlow的一个session运行[`tf.Session().run(tensor_name)`]之后的结果又是一个ndarray: 

     ```python
     >>> import tensorflow as tf
     >>> import numpy as np
     >>>
     >>> array1 = np.array([[1,2,3,4,5],[2,3,4,5,6]])
     >>> print(array1.shape)
     # Output: 
     (2, 5)
     >>> tensor1 = tf.constant(array1)
     >>> print(tensor1)
     Tensor("Const:0", shape=(2, 5), dtype=int32)
     >>> with tf.Session() as sess:
     ...     tensor1_value = sess.run(tensor1)
     ...     print(tensor1_value)
     ...     print(type(tensor1_value))
     # Output: 
     [[1 2 3 4 5]
      [2 3 4 5 6]]
     <class 'numpy.ndarray'>
     
     >>> print(array1[0])
     [1 2 3 4 5]
     >>> print(tensor1[0])
     Tensor("strided_slice:0", shape=(5,), dtype=int32)
     ```

   - shape和维度：从上面的代码也可以看出来，在下标访问和shape上，二者也是一致的。

## Tensor操作

1. `tf.reshape` reshape操作和 numpy 的操作基本是一样的

```python
>>> import tensorflow as tf
>>> import numpy as np
>>>
>>> array1 = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
>>> tensor1 = tf.constant(array1)
>>> print(tensor1)
# Output: 
Tensor("Const_2:0", shape=(2, 5), dtype=int32)
>>> tensor1 = tf.reshape(tensor1, (1, 2, 5))
>>> print(tensor1)
# Output: 
Tensor("Reshape_3:0", shape=(1, 2, 5), dtype=int32)

```

2. `expand_dims` 扩展维度，可以形象理解成给指定的维度加中括号，可以参见上面讲解维度的例子，参数`axis`为几，就是给对应的维度加上一个括号，如为0的时候就是给tensor自己加一个括号，为1的时候就是第一维，为2的时候就是给第二维（也就是第一维的每个切片中的每个元素），这样操作的结果就是给tensor增加了一个维度
3.  To be continued
   - 可能会专门写一个文档来讲tensor操作

## Tensorflow中的一些概念

1. 图(graph) 是TensorFlow中一些操作(operation)和tensor的集合，下面是一个简单的例子：

   ```python
   >>> import tensorflow as tf
   >>>
   >>> x = tf.constant([1, 2, 4, 5, 6], dtype='float32', name='x')
   >>> w = tf.constant([0.9, 1, 1.1, 0.9, 0.1], dtype='float32', name='weight')
   >>> b = tf.constant([0.01, 0.03, -0.02, 0.05, 0], dtype='float32', name='bias')
   >>>
   >>> y = x * w + b
   >>>
   >>> with tf.Session() as sess:
   ...     y_ret = sess.run(y)
   ...     print(y_ret)
   ...
   [0.90999997 2.03       4.38       4.55       0.6       ]
   ```

   上面是一个简单的线性操作，即`y = weight * x + bias`, 涉及了三个tensor和两个操作，可以得到下面的图：

   ![tensorboard](https://raw.githubusercontent.com/BridgeMia/Web-learning-of-Mia/master/pictures/tensor_graph.PNG)

   在后面更加复杂的TensorFlow代码中，如复杂的神经网络模型，我们可以利用tensorflow的图来直观得展现我们的模型的结构。


2. tf.Session()和TensorFlow的图

   - **没有值的graph**: 在一个图中，所有的tensor都是没有值的，正如最前面提到的tensor的基本信息中，在打印一个tensor的时候，我们只能知道这个tensor的`name`, `shape`, `dtype`, 我们可以认为在一个图中，tensor只包含了这些信息（其实还有其他的信息），为了获取这个tensor的值，我们就需要tf.Session()

   - tf.Session()是TensorFlow中的一个会话，它允许运行当前的图或者图的一部分，可以用一个客服的例子来理解：

     ```python
     import tensorflow as tf
     
     # ------------------------------------------------------------------------ #
     # Graph start
     x = tf.constant([1, 2, 4, 5, 6], dtype='float32', name='x')
     w = tf.constant([0.9, 1, 1.1, 0.9, 0.1], dtype='float32', name='weight')
     b = tf.constant([0.01, 0.03, -0.02, 0.05, 0], dtype='float32', name='bias')
     
     y = x * w + b
     # Graph end
     # ------------------------------------------------------------------------ #
     
     
     with tf.Session() as sess:
         # In the session, the agent only knows information about the above graph
     
         # query: What is the value of x?
         x_ret = sess.run(x)
         print(x_ret)
     
         # query: What is the value of y?
         y_ret = sess.run(y)
         print(y_ret)
     
         # query: What is the value of w * x?
         w_mul_x_ret = sess.run(w * x)
         print(w_mul_x_ret)
     
     # Output: 
     [1. 2. 4. 5. 6.]
     [0.90999997 2.03       4.38       4.55       0.6       ]
     [0.9 2.  4.4 4.5 0.6]
     ```

     在这个会话中，我们的agent（客服）只知道图(graph)中有关的信息，现在我们去和这个客服交互，就是在这个会话中依次运行图的某一部分，就能知道相应的值，当然你也可以每次询问都打开一个会话，但是当会话没有开启的时候，客服是没有办法回答你的，因此你也没有办法知道tensor的值。

3. tensorboard：可视化展现graph

   在上面的代码中，我们最后加上：

   ```python
   writer=tf.summary.FileWriter('logs', tf.get_default_graph())
   
   writer.close()
   ```

   就能把当前的图的信息储存在/logs/目录下，在终端（我测试的时候用的是pycharm中带的终端）中运行：

   ```shell
   tensorboard --logdir logs
   ```

   这个时候回返回一个类似的信息：

   ```shell
   TensorBoard 1.12.0 at http://EW620PC0JQX7L:6006 (Press CTRL+C to quit)
   ```

   但是这个链接可能打不卡，没关系，我们在浏览器中输入（如果你的代码是在本地运行的）

   ```http://localhost:6006```

   就能看到可视化的TensorFlow的图了。