#  margin-softmax loss

https://arxiv.org/pdf/1801.05599.pdf

## softmax loss

从softmax开始

softmax可以看成是一个压缩向量的函数，即对一个向量$$V$$，有：
$$
V = [V_i] \\
softmax(V) = [\frac{e^{V_i}}{\sum_j^C e^{V_j}}]
$$
其中$$C$$是向量的长度，也是总的类别数目。
$$
\mathcal{L}_S = -\frac{1}{n} \sum_{i=1}^{n} log \frac{e^{W^T_{y_i}\mathbf{f}_i}}{\sum_{j=1}^c e^{W^T_j \mathbf{f}_i}} \\
=-\frac{1}{n} log \frac{e^{\Arrowvert W_{y_i}\Arrowvert \Arrowvert\mathbf{f}_i \Arrowvert cos(\theta_{y_i})}}{e^{\Arrowvert W_{j}\Arrowvert \Arrowvert\mathbf{f}_i \Arrowvert cos(\theta_j)}}
$$
$$\mathbf{f_i}$$ 是最后一层全连接的输入，对于输入网络的一条数据，应该是一个一维的向量(在后面可以看做是我们的encoder编码好的向量)，$$W$$是线性分类器的权重矩阵，$$W^T_{y_i}\mathbf{f}_i$$就是对标签$$y_i$$(是一个数字)的logit，这里可以展开解释一下：

对于一个n分类的分类器，输入$$\mathbf{f}$$, 分类器的输出是这样的一个向量：
$$
\mathbf{p} = softmax(\mathbf{f}W)
\\
W = (\mathbf{c}_1, \mathbf{c}_2, \cdots, \mathbf{c}_n )
$$
这样输出的是一个概率分布$$\mathbf{p}$$, 是一个n维的向量，每个维度代表输入在属于这个分类的概率，由上面可以把概率分布改写为：
$$
\mathbf{p} = softmax(\mathbf{f} \otimes \mathbf{c}_1, \mathbf{f} \otimes \mathbf{c}_2, 
\cdots, \mathbf{f} \otimes \mathbf{c}_n)
$$
对于标签t，损失函数就是交叉熵，也就是
$$
-log \,\mathbf{p}_t = - log \frac{e^{\mathbf{f} \otimes \mathbf{c}_t}}{\sum_{i=1}^n 
e^{\mathbf{f} \otimes \mathbf{c}_i}}
$$
这里的$$\otimes$$表示向量内积：
$$
\mathbf{x} \otimes \mathbf{y} = \sum_i^m x_iy_i
\\
cos(\mathbf{x}, \mathbf{y}) = \frac{\sum_i^m x_iy_i}{\Arrowvert \mathbf{x} \Arrowvert \Arrowvert \mathbf{y} \Arrowvert}
$$
所以上面loss可以写成上面那样的形式

假设上面的两个向量都是归一化的，那么余弦距离就等于内积

## am-softmax

am-softmax的目的是找到一个比余弦函数更严格的距离表示方法来代替它， 就可以实现margin-softmax了，比如在原文中，使用的是
$$
\varphi(\theta) = s (cos(\theta) - m)
$$
