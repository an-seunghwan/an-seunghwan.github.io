---
title: "Byte Pair Encoding (1)"
excerpt: "BPE의 개념"
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-06 22:00:00 -0000
categories: 
  - NLP
tags:
  - OOV
  - tokenizer
  - 전처리
---
	톺아보기 시리즈 (톺아보다: 샅샅이 틈이 있는 곳마다 모조리 더듬어 뒤지면서 살피다)

> 이 포스팅은 OOV(out-of-vocabulary) 문제의 해결과 관련한 내용, 논문, tokenizer 등을 다루는 [OOV 톺아보기 시리즈](https://an-seunghwan.github.io/oov-top-a-bogi/)의 1편입니다.

## What is OOV?

corpus에 기존에 존재하지 않은 단어를 뜻하며, 이러한 OOV가 등장하게 되면 문제 해결이 어려워진다. 이러한 문제를 해결하기 위해 등장한 알고리즘이 단어의 segmentation를 통해서 subword를 만들고 이를 이용하는 **Byte Pair Encoding**이다.

여기서 Byte는 한 음절을 의미한다고 이해하면 좋다.

## What is BPE(Byte Pair Encoding)?

### setup
```python
import re, collections
from pprint import pprint
```

### 단어 사전

corpus에서 주어진 `단어 : 빈도수` 사전이 다음과 같다고 가정하자.
```python
vocab = {'장 난 꾸 러 기 </w>': 5,
         '잠 꾸 러 기 </w>': 6,
         '장 난 감 </w>': 10,
         '잠 수 </w>': 3,
         '욕 심 </w>': 4}
```
**띄어쓰기로 각 음절을 구분해 놓은 것에 주목해야 한다.** 즉, 처음으로 단어 사전을 구성할 때는 각 음절이 별도로 취급되어야 한다. 또한, 위의 단어 사전은 빈도수를 얻기 위해 사용하는 단어 사전이다.

초기 단어 사전의 형태는 다음과 같다(이를 단어 사전의 segmentation이라 하자).
```python
def dict_segmentation(vocab):
    initial_vocab = set()
    for word in vocab.keys():
        symbols = word.split()
        initial_vocab.update(symbols)
    pprint(initial_vocab)

dict_segmentation(vocab)
```
```
{'장', '욕', '감', '난', '수', '심', '러', '잠', '꾸', '기'}
```
이 초기의 각 byte(음절)로 구성된 단어 사전을 update하여 최종 단어 사전을 얻는다.

### Byte 조합의 빈도수 계산 함수

```python
def get_stats(vocab):
    pairs = collections.defaultdict(int) # 값을 저장할 빈 dict
    for word, freq in vocab.items():
        symbols = word.split() # 띄어쓰기를 기준으로 분할
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs
```
초기 단어 사전에 대해 어떤 결과가 나오는 지 살펴보자.
```python
pprint(get_stats(vocab))
```
```
defaultdict(<class 'int'>,
            {('감', '</w>'): 10,
             ('기', '</w>'): 11,
             ('꾸', '러'): 11,
             ('난', '감'): 10,
             ('난', '꾸'): 5,
             ('러', '기'): 11,
             ('수', '</w>'): 3,
             ('심', '</w>'): 4,
             ('욕', '심'): 4,
             ('잠', '꾸'): 6,
             ('잠', '수'): 3,
             ('장', '난'): 15})
```

### 빈도수를 기준으로 Byte를 Pairing!

가장 높은 빈도수의 Byte 쌍을 추출한다.
```python
pairs = get_stats(vocab)
best = max(pairs, key=pairs.get)
print(best)
```
```
('장', '난')
```
`('장', '난')` byte 쌍이 빈도수가 15로 가장 높다. 이제 다음의 함수를 이용해 앞의 byte를 pairing한다.

* merge_vocab의 detail한 해석!

```python
def merge_vocab(pair, v_in):
	# pair: 가장 높은 빈도의 byte pair(입력 받는 인자)
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
	    # 기존의 단어 사전에서 가장 높은 빈도수의 byte pair와 동일한 문자열을 교체
	    # p: pattern
		# repl: 바꾸려는 문자열
        # string: 바꾸는 대상이 되는 문자열
        w_out = p.sub(repl=''.join(pair), string=word)
        v_out[w_out] = v_in[word]
    return v_out
```
```python
vocab = merge_vocab(best, vocab)
pprint(vocab)
```
```
{'욕 심 </w>': 4,
 '잠 꾸 러 기 </w>': 6,
 '잠 수 </w>': 3,
 '장난 감 </w>': 10,
 '장난 꾸 러 기 </w>': 5}
```
가장 빈도수가 높았던 byte 쌍인 `('장', '난')` 가 하나의 음절로 묶인 것을 볼 수 있다. 이제 어떠한 단어 사전의 segmentation을 확인하자.
```python
dict_segmentation(vocab)
```
```
{'장난', '욕', '감', '수', '심', '러', '잠', '꾸', '기'}
```

### end-to-end flow

단어 사전이 어떻게 변해가는지 전체적으로 확인해보자.
```python
# BPE 수행 최대 횟수(병합 횟수)
MAX_MERGES = 20
count = 0
while(count <= MAX_MERGES):
    pairs = get_stats(vocab)
    if len(pairs) == 0:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print("step {}".format(count+1))
    print("Most frequent byte pair: {}".format(best))
    dict_segmentation(vocab)
    count += 1
```
```
step 1
Most frequent byte pair: ('장', '난')
{'장난', '욕', '</w>', '감', '수', '심', '러', '잠', '꾸', '기'}
step 2
Most frequent byte pair: ('꾸', '러')
{'장난', '꾸러', '욕', '</w>', '감', '수', '심', '잠', '기'}
step 3
Most frequent byte pair: ('꾸러', '기')
{'장난', '욕', '</w>', '감', '꾸러기', '수', '심', '잠'}
step 4
Most frequent byte pair: ('꾸러기', '</w>')
{'장난', '꾸러기</w>', '욕', '</w>', '감', '수', '심', '잠'}
step 5
Most frequent byte pair: ('장난', '감')
{'장난', '꾸러기</w>', '욕', '</w>', '장난감', '수', '심', '잠'}
step 6
Most frequent byte pair: ('장난감', '</w>')
{'장난', '꾸러기</w>', '욕', '</w>', '장난감</w>', '수', '심', '잠'}
step 7
Most frequent byte pair: ('잠', '꾸러기</w>')
{'장난', '꾸러기</w>', '잠꾸러기</w>', '</w>', '욕', '장난감</w>', '수', '심', '잠'}
step 8
Most frequent byte pair: ('장난', '꾸러기</w>')
{'장난꾸러기</w>', '잠꾸러기</w>', '</w>', '욕', '장난감</w>', '수', '심', '잠'}
step 9
Most frequent byte pair: ('욕', '심')
{'장난꾸러기</w>', '잠꾸러기</w>', '</w>', '장난감</w>', '욕심', '수', '잠'}
step 10
Most frequent byte pair: ('욕심', '</w>')
{'장난꾸러기</w>', '잠꾸러기</w>', '</w>', '욕심</w>', '장난감</w>', '수', '잠'}
step 11
Most frequent byte pair: ('잠', '수')
{'장난꾸러기</w>', '잠꾸러기</w>', '</w>', '욕심</w>', '장난감</w>', '잠수'}
step 12
Most frequent byte pair: ('잠수', '</w>')
{'장난꾸러기</w>', '잠꾸러기</w>', '욕심</w>', '장난감</w>', '잠수</w>'}
```

## reference
Sennrich, R., Haddow, B., & Birch, A. (2015). Neural machine translation of rare words with subword units. _arXiv preprint arXiv:1508.07909_.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTcxNTYxODUwMl19
-->