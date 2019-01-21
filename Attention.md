# Attention

## Attention机制

attention机制的输入是Query(q), key-value pair(k, v), 根据q和k来计算attention, 然后把attention加到v上。

例如对两个句子做attention, 这里的q可以是其中一个句子，k和v相同，是另外一个句子。



## Dot-Production Attention

基于点积的attention, 两个element之间的attention用点积来计算

向量点积的公式：
$$
\vec A \cdot \vec B = \sum_{i}^n a_i b_i
$$
可以用`tf.tensordot`来实现：

```python
a = tf.constant(np.arange(7, 13, dtype=np.int32))
b = tf.constant(np.arange(19, 25, dtype=np.int32))

dot_ret = tf.tensordot(a, b, axis=1)

print(tf.Session().run(dot_ret)
# Output: 
1243

```



### Scaled Dot-Production Attention

```python
class ScaledDotProductAttention:
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(x[0],x[1],axes=[2, 2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn
```

在我们的问题中，暂时不考虑mask, k和v是相同的，并且要求输入的两个问题的位置可以互换，因此可以简化成下面的版本：

```python
class MutualScaledDotProductAttention:
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, v):
        # Attention matrix: A
        # A_ij is the attention of word_i in q1 to word_j in q2
        attn = Lambda(lambda x: K.batch_dot(x[0],x[1],axes=[2, 2])/self.temper)([q, v])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        v_attn = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return v_attn
```

scale的目的是让dot之后的结果还是在一个比较小的范围内，否则softmax的时候会出现一个是1其他全是0的情况，因为softmax函数用的是幂函数：
$$
\mathbf{X} = [x_i] \\
softmax(\mathbf{X}) = [\frac{e^{x_i}}{\sum_i{e^{x_i}}}]
$$
假设我们的输入是两个$$30 \times 256$$ 的矩阵，即句子长度为30, 词向量的维度为256，得到的attention矩阵
$$
\mathbf{A} = [a_{i,j}]
$$
其中$$a_{i,j}$$表示第一个句子的第i个词对第二个句子中的第j个词的attention

经过softmax之后，对行进行了归一化，即有$$\sum_j{a_{i,j}} = 1$$, attention权重为第二个句子中的每个词的attention权重求和，最后返回一个带attention权重的句子。

### Attention的作用

1. 在bidaf中输入到attention部分的已经经过了LSTM层提取特征，所以这一步可以看做是滤波，把不重要的词语滤掉，留下重要的词语。
2. Attention矩阵本身也反应了两个句子之间的相似度，如果我们分别对Attention矩阵按行和按列求和，就可以得到两个句子之间互相的Attention.