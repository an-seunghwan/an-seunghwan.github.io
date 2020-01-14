# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:19:25 2020

@author: mayson
"""


#%%
import pandas as pd
from progressbar import progressbar
from pprint import pprint
import csv

#%%
import sys
sys.path.insert(0, r'C:\Users\dpelt\Desktop\Mayson\UOS_graduate\translator\src')
from BPESegmenter import BPESegmenter

#%%
'''
데이터 불러오기
'''
DATA_PATH = r'C:\Users\dpelt\Desktop\Mayson\UOS_graduate\translator'
translation_path = r'\translation_data.csv'

#%%
# =============================================================================
# 번역 데이터
# =============================================================================
sample_size = 5000
trans_data = pd.read_csv(DATA_PATH + translation_path, 
                         encoding='cp949', 
                         header=0).iloc[:sample_size+100]
print(trans_data.iloc[0])
train_data = list(trans_data['원문'].iloc[:sample_size])
train_data.extend(list(trans_data['번역문'].iloc[:sample_size]))

test_data = list(trans_data['원문'].iloc[-100:])
test_data.extend(list(trans_data['번역문'].iloc[-100:]))

#%%
'''
주어진 corpus에 대해 BPE를 이용해 단어 사전 구축
'''
MAX_MERGE = 6400 # parameter
bpe = BPESegmenter(MAX_MERGE, 
                   reduction=0.8) # vocab reduction은 80%로 설정
bpe.train(train_data)

#%%
vocab = bpe.vocab
print('초기 단어 사전의 크기: {}'.format(len(vocab)))
pprint(list(vocab.items())[100:110])
print(bpe.vocab_score)

#%%
'''
단어 사전 저장
'''
#save_path = DATA_PATH + r'\data_configs.json'
#bpe.save(save_path)

#%%
'''
단어 사전 불러오기
'''
#load_path = DATA_PATH + r'\data_configs.json'
#bpe.load(load_path)

#%%
'''
reduction을 진행하기 전, tokenize된 corpus를 만든다.
tokenizing과 동시에 self.vocab_score가 만들어진다.
'''
tokenized_result = []
for i in progressbar(range(len(train_data))):
    tokenized_result.append(bpe.tokenize(train_data[i]))
print(tokenized_result[:10])

#%%
vocab_score = bpe.vocab_score
print(len(vocab_score))
print(list(vocab_score.items())[:10])

#%%
'''
비교를 위해 초기 단어 사전에 대한 tokenized corpus를 저장
'''
temp_tokenized = pd.concat([pd.DataFrame(tokenized_result[:sample_size]), pd.DataFrame(tokenized_result[sample_size:])], axis = 1)
temp_tokenized.to_csv(DATA_PATH + '/result/init_tokenized_result_temp.csv', 
                      sep=',',
                      index=False,
                      header=False,
                      na_rep='NaN',
                      encoding='cp949')
    
#with open(DATA_PATH + '/init_tokenized_result.csv', 'w', newline='') as f:
#    csv_writer = csv.writer(f, delimiter=',')
#    for row in temp_tokenized:
#        csv_writer.writerow([row])

#%%
'''
단어 사전 축소
'''
bpe.vocab_reduction()
vocab = bpe.vocab
print('축소 후 단어 사전의 크기: {}'.format(len(vocab)))

#%%
'''
단어 사전 축소 후 tokenize
-> 단어 축소 이후에는 self.vocab_score를 사용하지 못함
'''
reduction_tokenized_result = []
for i in progressbar(range(len(train_data))):
    reduction_tokenized_result.append(bpe.tokenize(train_data[i]))
pprint(reduction_tokenized_result[:10])

#%%
#vocab_score = bpe.vocab_score
#print(len(vocab_score))
#print(list(vocab_score.items())[:10])

#%%
temp_reduction_tokenized = pd.concat([pd.DataFrame(reduction_tokenized_result[:sample_size]), pd.DataFrame(reduction_tokenized_result[sample_size:])], axis = 1)
temp_reduction_tokenized.to_csv(DATA_PATH + '/result/reduction_tokenized_result_temp.csv', 
                                sep=',',
                                index=False,
                                header=False,
                                na_rep='NaN',
                                encoding='cp949')

#%%
'''
testing
'''
test_tokenized = []
for i in progressbar(range(len(test_data))):    
    test_tokenized.append(bpe.tokenize(test_data[i].replace(' ', '_') + '_'))
pprint(test_tokenized[:10])

#%%
temp_tokenized = pd.concat([pd.DataFrame(test_data[:100]), pd.DataFrame(test_tokenized[:100]), pd.DataFrame(test_tokenized[100:])], axis = 1)
temp_tokenized.to_csv(DATA_PATH + '/result/test_tokenized_result_temp.csv', 
                      sep=',',
                      index=False,
                      header=False,
                      na_rep='NaN',
                      encoding='cp949')

#%%
temp_vocab = sorted(vocab.items(), key=lambda x: -x[1])
with open(DATA_PATH + r'\result\vocab_temp.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',')
    for row in temp_vocab:
        csv_writer.writerow(row)
        
#%%






























