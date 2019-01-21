# BIDAF

BI-DIRECTIONAL ATTENTION ATTENTION FLOW

ref:  [BI-DIRECTIONAL ATTENTION FLOW FOR MACHINE COMPREHENSION, ICLR 2017](https://arxiv.org/pdf/1611.01603.pdf)

## 模型的结构

1. Char Embedding
2. Word Embedding
3. Contextual Embedding 
4. Attention Flow
5. Modeling
6. Output



## Contextual Embedding

以Word Embedding的结果为例，结果是一个$$ \mathbf{X} \in \mathbb{R} ^{d \times T} $$ 的矩阵，其中$$d$$ 是`word2vec`的size, $$T$$ 是句子的长度。经过一个双向的`LSTM`来提取单词之间的相互关系，正向和反向的`LSTM`的的结果经过词向量维度上的拼接得到一个 $$ \mathbf{X} \in \mathbb{R} ^{2d \times T}  $$的矩阵。仿照原文的写法，两个问题在这一层的输出分别记为 $$\mathbf{H}$$ 和 $$\mathbf{U} $$ .

对于相同句子的比较，这我们的原始输入是Q1和Q2, 所以这里没有原文中Query和Context的区分。

## Attention Flow

1. Similarity Matrix

$$
\mathbf{S} \in \mathbb{R} ^{T \times T}
$$

​	where
$$
\mathbf{S}_{i, j} = \alpha (\mathbf{H}_{:i}, \mathbf{U}_{:j}) = \alpha(\mathbf{h, u})
$$


​	计算了Q1和Q2经过了上面的Embedding之后输出结果的相似度，也就是对于两个句子中的每一个向量通过	一个可以训练的方法$$\alpha$$计算相似度。

原文中使用的方法是
$$
\alpha (\mathbf{h, u}) = \mathbf{w}_{(s)} ^ \intercal [\mathbf{h;u;h \circ u}]
$$
$$\mathbf{w}_{(s)} \in \mathbb{R} ^ {6d}$$ 是一个可以训练的权重矩阵，[]表示其中的三个向量在词向量维度上的拼接，$$ \circ $$ 表示元素矩阵乘法（元素乘元素）。

相似度计算也可以直接用两个向量点乘。

2. Context-to-query Attention：

   表示Q2中哪个词对Q1最重要

$$
\mathbf{a}_{i} = softmax(\mathbf{S}_{i:}) \in \mathbb{R} ^ T
$$

​	我们知道，$$\mathbf{s}_{i,j}$$ 表示的是 Q1中的第i个词对Q2中的第j个词的attention, 所以$$\mathbf{S}_{i:}$$ 就表示Q1这样的attention weight构成的一个序列（向量），用softmax来保证$$ \sum \mathbf{a}_{i,j} = 1$$. 

接下来把attention weight乘上：
$$
\tilde{\mathbf{U}}_{i} = \sum_j \mathbf{a}_{i,j} \mathbf{U}_{:j} \in \mathbb{R} ^ {2d}
$$
​	那么$$\tilde{\mathbf{U}}_{i}$$ 就是Q2的每个词的词向量的加权平均，权重为这个词语对Q1中第i个词的attention

​	由$$\tilde{\mathbf{U}}_{i}$$ 构成的序列$$\tilde{\mathbf{U}}$$ 就是对整个Q1的带attention权重的Q2的向量表示，换句话说，对于每个Q1中的词语，Q2都有一种表示方法$$\tilde{\mathbf{U}}_{i}$$ .

3. Query-to-context Attention:

表示Q1中哪个词对Q2最重要：
$$
\mathbf{b} = softmax(max_{col}(\mathbf{S})) \in \mathbb{R} ^ T
$$
$$\mathbf{b}$$ 就表示Q1中的第i个词对Q2中所有词的最大attention
$$
\tilde{\mathbf{h}} = \sum_i \mathbf{b}_i \mathbf{H}_{:i} \in \mathbb{R} ^ {2d}
$$
这样得到的向量表示了对Q1中的每个词的加权平均。

这样得到的$$\tilde{\mathbf{h}}$$ 跟i没有关系，操作T次得到的矩阵$$\tilde{\mathbf{H}} \in \mathbb{R} ^ {2d \times T}$$ ，也就是说$$\tilde{\mathbf{H}}$$是由T个同样的$$\tilde{\mathbf{h}}$$构成的。