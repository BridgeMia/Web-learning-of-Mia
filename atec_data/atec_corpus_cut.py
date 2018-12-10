import bmtk
import pandas as pd
import requests
from tqdm import tqdm
import threading
import numpy as np

corpus = bmtk.load_list(r'data/ATEC_corpus/questions.txt')
currently_processed = bmtk.loadCSV(r'data/ATEC_corpus/entity_linking_cut_ret.txt')
processed_questions = np.array(currently_processed)[:,0]
processed_questions = {x:0 for x in processed_questions}

corpus = [x for x in corpus if x not in processed_questions.keys()]
api_key = 'ljqljqljq'
ret = currently_processed

for i in tqdm(range(len(corpus))):
    q = corpus[i]
    url = 'http://shuyantech.com/api/entitylinking/cutsegment?q=%s&apikey=%s'%(q,api_key)
    cut_ret = requests.get(url).json()
    cut_q = '-'.join(cut_ret['cuts'])
    entities = '-'.join([x[1] for x in cut_ret['entities']])
    ret.append([q, cut_q, entities])
    if i%1000 == 1:
        print('save current process ret')
        bmtk.saveCSV(ret, r'data/ATEC_corpus/entity_linking_cut_ret.txt')

bmtk.saveCSV(ret, r'data/ATEC_corpus/entity_linking_cut_ret.txt')