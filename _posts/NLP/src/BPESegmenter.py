# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:02:53 2020

@author: mayson
"""


#%%
# =============================================================================
# what to do
# =============================================================================
# 1. docstring 추가

#%%
import collections
from copy import deepcopy
import json

#%%
class BPESegmenter:
    
    def __init__(self, merge_num, reduction=0, verbose=True):
        '''
        merge_num: BPE 수행 최대 횟수(병합 횟수)    
        reduction: must be 0 to 1
        '''
        self.merge_num = merge_num
        self.vocab = {} # vocabulary of subwords
        self.vocab_score = {}
        self.max_length = 0
        self.total_freq = 0
        self.reduction = reduction
        self.verbose = verbose
        
    def train(self, text):
        if self.verbose:
            print('Begin subwords scanning', end='', flush=True)
        
        subwords = self._build_subwords(text)
        
        if self.verbose:
            print('\rSubwords scanning terminated', flush=True)
        
        self.vocab = self._build_vocab(subwords)
        
    def _build_subwords(self, corpus):
        seed_vocab = collections.Counter((word.replace('_', '') for sent in corpus for word in sent.split() if word))
        return {' '.join(word) + ' _' : freq for word, freq in seed_vocab.items() if word}

    def _build_vocab(self, subwords):
        def get_stats(freq_vocab):
            pairs = collections.defaultdict(int) 
            for word, freq in freq_vocab.items():
                symbols = word.split()
                for i in range(len(symbols)-1):
                    pairs[symbols[i], symbols[i+1]] += freq
            return pairs        

        def merge_vocab(pair, v_in):
            v_out = {}
            bigram = ' '.join(pair)
            replacer = ''.join(pair)
            for word, freq in v_in.items():
        #        v_out[word] = freq
                w_out = word.replace(bigram, replacer)
                v_out[w_out] = freq
            return v_out

        for i in range(self.merge_num):
            pairs = get_stats(subwords)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            subwords = merge_vocab(best, subwords)
            
            if self.verbose:
                print('\rTraining BPESegmenter {} / {}'.format(i+1, self.merge_num), end='', flush=True)
        
        if self.verbose:
            print('\rTraining BPESegmenter was done'.format(' '*40), flush=True)

        vocab = {}
        for word, freq in subwords.items():
            for subword in word.split():
                vocab[subword] = vocab.get(subword, 0) + freq
        
        self.max_length = max([len(x) for x in list(vocab.keys())])
        
        self.total_freq = float(sum(list(vocab.values())))
        
        return vocab

    def tokenize(self, sequence):
        return self._tokenize(sequence)
    
    def _tokenize(self, sequence):

        def get_subword_candidates(input_sequence):
            input_sequence = '_'.join(input_sequence.split()) + '_'
            candidates = []
            n = len(input_sequence)
            for start in range(n):
                for end in range(start+1, min(n, start+self.max_length)+1):
                    subword = input_sequence[start:end]
                    if subword[-1] == ' ':
                        subword = subword.replace(' ', '_')
                    if not subword in self.vocab:
                        continue
                    candidates.append((subword, start, end, end-start, self.vocab.get(subword)))
            candidates = [x for x in candidates if x[0] != '_']
            return candidates   
        
        def get_max_candidate(candidates):
            matched = []
            topk_sequence = []
            # 굳이 sort할 필요는 없으나 sort 기준 수정을 위해 유지
            segments = sorted(candidates, key=lambda x:(-x[3], x[1])) 
            topk_segments = sorted(candidates, key=lambda x:(-x[3], x[1]))
            
            for k in range(len(topk_segments)): # 모든 경우에 대해 tokenized 결과와 그 score를 계산
                matched = []
                score = 1
                temp_segments = deepcopy(segments)
                string, start, end, length, freq = topk_segments[k]
                matched.append((string, start, end, length, freq))
                score *= freq / self.total_freq
                temp_segments.remove(topk_segments[k])
                removals = []
                for i, (_, s, e, _, _) in enumerate(temp_segments):
                    if not (e <= start or end <= s): 
                        removals.append(i)
                for i in reversed(removals):
                    del temp_segments[i]
                    
                while temp_segments:
                    string, start, end, length, freq = temp_segments.pop(0)
                    matched.append((string, start, end, length, freq))
                    score *= freq / self.total_freq
                    removals = []
                    for i, (_, s, e, _, _) in enumerate(temp_segments):
                        if not (e <= start or end <= s): 
                            removals.append(i)
                    for i in reversed(removals):
                        del temp_segments[i]
            
                topk_sequence.append((' '.join([x[0] for x in sorted(matched, key=lambda x: x[1])]), score))
            
            return max(topk_sequence, key=lambda x: x[1])
        
        max_candidate = get_max_candidate(get_subword_candidates(sequence))
        for subword in max_candidate[0].split():
            self.vocab_score[subword] = self.vocab_score.get(subword, 0) + max_candidate[1]
        
        not_used_subwords = [x for x in self.vocab.keys() if x not in self.vocab_score.keys()]
        for n in not_used_subwords:
            self.vocab_score[n] = 0
        
        return max_candidate[0]
    
    def vocab_reduction(self):
        vocab_size = round(self.reduction * len(self.vocab))
        temp_vocab = [word for i, (word, freq) in enumerate(sorted(self.vocab_score.items(), key=lambda x: -x[1])) if i < vocab_size or len(word.replace('_', '').replace('.', '')) == 1]
        self.vocab = {word : freq for word, freq in self.vocab.items() if word in temp_vocab}
        
        self.total_freq = float(sum(list(self.vocab.values())))
        
    def save(self, save_path):
        data_configs = {}
        data_configs['vocab'] = self.vocab
        data_configs['vocab_score'] = self.vocab_score
        data_configs['max_length'] = self.max_length
    
        json.dump(data_configs, open(save_path, 'w'))
        
    def load(self, load_path):
        data_configs = json.load(open(load_path, 'r'))
        self.vocab = data_configs['vocab']
        self.vocab_score = data_configs['vocab_score']
        self.max_length = data_configs['max_length']

#%%







