import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.models import Model

# Scaled dot-production attention
"""
Given q, k, v, the attention weight is square(q dot k),
and return weighted v according to the attention
"""
data = np.random.random((20, 30, 256))
feed_q1 = data[:10, :, :]
feed_q2 = data[10:, :, :]


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



q1_in = Input(shape=(5, 10))
q2_in = Input(shape=(5, 10))

attention = MutualScaledDotProductAttention(d_model=10)
q1_attn = Lambda(lambda x: attention(x[1], x[0]))([q1_in, q2_in])
q2_attn = Lambda(lambda x: attention(x[0], x[1]))([q1_in, q2_in])

attn = Lambda(lambda x: K.batch_dot(x[0],x[1],axes=[2, 2])/np.sqrt(10))([q1_in, q2_in])
attn = Activation('softmax')(attn)

model = Model((q1_in, q2_in), (q1_attn, q2_attn, attn))

if __name__ == '__main__':
    x1 = np.arange(0, 50).reshape((1, 5, 10))/50
    x2 = np.arange(50, 100).reshape((1, 5, 10))/50

    model.compile(optimizer='adam', loss=lambda y_p,y_pred:y_p)
    ret1 = model.predict([x1, x2])
    ret2 = model.predict([x2, x1])

    attn_matrix = ret1[2]
    q1_to_q2 = ret1[0]
    print(q1_to_q2)
    print(attn_matrix[0])
