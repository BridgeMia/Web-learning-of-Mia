import pandas as pd
from gensim.models.word2vec import Word2Vec
import numpy as np
from nlpops import Char2ID

corpus1 = ['帮我-关掉-借呗-可以-吗', '蚂蚁-借呗-提前-还-了-分期，到期-后-还要-还-这期-吗']
corpus2 = ['天猫店-都-能用-花呗-吗', '我-用-花呗-买东西，隔天-就-还款-了。为什么-现在-又-扣款']
corpus3 = ['天猫店-都-能用-花呗-吗', '我-用-花呗-买东西，隔天-就-还款-了。为什么-现在-又-扣款']

corpus1c = ['帮我关掉借呗可以吗', '蚂蚁借呗提前还了分期，到期后还要还这期吗']
corpus2c = ['天猫店都能用花呗吗', '我用花呗买东西，隔天就还款了。为什么现在又扣款']
corpus3c = ['天猫店都能用花呗吗', '我用花呗买东西，隔天就还款了。为什么现在又扣款']

char2id = Char2ID.load(r'data/char2id_test.txt').char2id_dict

"""
w2v_mod_sents = [x.split('-') for x in corpus1] + [x.split('-') for x in corpus2]
w2v_mod = Word2Vec(sentences=w2v_mod_sents, size=256, window=3, min_count=1)
w2v_mod.save(r'data/w2v_test_mod')
"""

w2v_mod = Word2Vec.load(r'data/w2v_test_mod')

train = pd.DataFrame({'q1_cut': [corpus1[0]],
                      'q2_cut': [corpus1[1]],
                      'q1_chars': [[char2id[x] for x in corpus1c[0]]],
                      'q2_chars': [[char2id[x] for x in corpus1c[1]]],
                      'flag': [0]})

valid = pd.DataFrame({'q1_cut': [corpus2[0]],
                      'q2_cut': [corpus2[1]],
                      'q1_chars': [[char2id[x] for x in corpus2c[0]]],
                      'q2_chars': [[char2id[x] for x in corpus2c[1]]],
                      'flag': [0]})

test = pd.DataFrame({'q1_cut': [corpus3[0]],
                     'q2_cut': [corpus3[1]],
                     'q1_chars': [[char2id[x] for x in corpus2c[0]]],
                     'q2_chars': [[char2id[x] for x in corpus2c[1]]],
                     'flag': [0]})

data = [train, valid, test]
cols = ['q1_cut', 'q2_cut']

for p_data in data:
    for col in cols:
        p_data[col] = p_data[col].apply(lambda x: x.split('-'))
        p_data[col] = p_data[col].apply(lambda x: np.array([w2v_mod[z] for z in x]))

# Train shape: [length_of_train, 30, 256]
train_input = [np.zeros((len(train), 30, 256)) for _ in range(2)] + [np.zeros((len(train), 30)) for _ in range(2)]

for index, row in train.iterrows():
    uw = row['q1_cut'][0:30]
    vw = row['q2_cut'][0:30]
    uc = row['q1_chars'][0:30]
    vc = row['q2_chars'][0:30]
    train_input[0][index][0:len(uw)] = uw
    train_input[1][index][0:len(vw)] = vw
    train_input[2][index][0:len(uc)] = uc
    train_input[3][index][0:len(vc)] = vc

for x in train_input:
    print(x)
