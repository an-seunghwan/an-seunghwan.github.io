---
title: "Byte Pair Encoding (2)"
excerpt: "실제 데이터 적용하기"
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-07 16:00:00 -0000
categories: 
  - NLP
tags:
  - OOV
  - tokenizer
  - 전처리
---
	톺아보기 시리즈 (톺아보다: 샅샅이 틈이 있는 곳마다 모조리 더듬어 뒤지면서 살피다)

> 이 포스팅은 OOV(out-of-vocabulary) 문제의 해결과 관련한 내용, 논문, tokenizer 등을 다루는 [OOV 톺아보기 시리즈](https://an-seunghwan.github.io/oov-top-a-bogi/)의 3편입니다.

지난 포스팅들에서 BPE와 관련한 논문과 그 개념, 그리고 간단한 코드를 알아보았다. 이번 게시글에서는 실제 데이터에 적용하여 그 결과를 확인해보도록 하겠다. 

## 데이터 
* 네이버 영화 리뷰 데이터 'Naver sentiment movie corpus v1.0'
* 출처: [https://github.com/e9t/nsmc](https://github.com/e9t/nsmc)
* 본 글에서는 평가 데이터만을 예시로 사용한다.

## setup
```python
import pandas as pd
import re
from progressbar import progressbar
from konlpy.tag import Okt
from pprint import pprint

DATA_PATH = r'C:\Users\dpelt\Downloads\nsmc-master\nsmc-master'
```

## 파일 읽기
```python
data = pd.read_csv(DATA_PATH + '/' + FILE_NAME, 
                   header=0,
                   delimiter='\t',
                   quoting=3)
# 텍스트 데이터만을 따로 뽑아서 사용
text_data = data['document'][:1000]
text_data.head()
```
```
0                                                  굳 ㅋ
1                                 GDNTOPCLASSINTHECLUB
2               뭐야 이 평점들은.... 나쁘진 않지만 10점 짜리는 더더욱 아니잖아
3                     지루하지는 않은데 완전 막장임... 돈주고 보기에는....
4    3D만 아니었어도 별 다섯 개 줬을텐데.. 왜 3D로 나와서 제 심기를 불편하게 하죠??
Name: document, dtype: object
```

## 한글화 정제
```python
def clean_korean(sent):
    if type(sent) == str:
        h = re.compile('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]+')
        result = h.sub(' ', sent)
    else:
        result = ''
    return result

clean_text = []
for i in tqdm(range(len(text_data))):
    sent = clean_korean(text_data.iloc[i])
    if len(sent) and sent != ' ': # 비어있는 데이터가 아닌지 확인
        clean_text.append(sent)
```
```
100% (1000 of 1000) |####################| Elapsed Time: 0:00:00 Time:  0:00:00
```
```python
pprint(clean_text[:6])
```
```
['굳 ㅋ',
 '뭐야 이 평점들은  나쁘진 않지만  점 짜리는 더더욱 아니잖아',
 '지루하지는 않은데 완전 막장임  돈주고 보기에는 ',
 ' 만 아니었어도 별 다섯 개 줬을텐데  왜  로 나와서 제 심기를 불편하게 하죠 ',
 '음악이 주가 된  최고의 음악영화',
 '진정한 쓰레기']
```
 
## BPE 알고리즘

### 함수 정의
```python
def init_vocab(corpus):
    '''
    초기 단어 사전을 구축(단어 : 빈도수)
    corpus: 문장별 list
    '''
    subwords = []
    for sent in corpus:
        temp = [' '.join(word) + ' </w>' for word in sent.split()]
        subwords.extend(temp)
    
    vocab = collections.Counter(subwords)
    return vocab
    
def vocab_segmentation(vocab):
    '''
    전체 단어에 대해 segment하여 subwords로 구성된 단어 사전을 구축
    '''
    vocab_segment = set()
    for word in vocab.keys():
        symbols = word.split()
        vocab_segment.update(symbols)
    return list(vocab_segment)
    
def get_stats(vocab):
    '''
    각 byte pair의 빈도수를 계산
    '''
    pairs = collections.defaultdict(int) 
    for word, freq in vocab.items():
        symbols = word.split() 
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    '''
    빈도수가 높은 byte pair로 기존의 vocab을 update
    '''
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(repl=''.join(pair), string=word)
        v_out[w_out] = v_in[word]
    return v_out
```

### step 1. 초기 단어 사전 구축
```python
vocab = init_vocab(clean_text)
subwords_dict = vocab_segmentation(vocab)
print('subwords 사전의 개수: {}'.format(len(subwords_dict)))
print(subwords_dict[:100])
```
```
subwords 사전의 개수: 991
['간', '베', '링', '적', '권', '딘', 'ㄹ', '펐', '룬', '뀐', '쒀', '잤', '이', '행', '탕', '듣', '맣', '해', '픽', '돈', '대', '젖', '흡', '숙', '똥', '쟎', '폭', '왜', '댄', '신', '렀', '낫', 'ㅗ', '눈', '빡', '처', '황', '굉', '학', '즌', '댓', '패', '윌', '근', '넼', '좔', '쟁', '주', '쭈', 'ㅜ', '매', '획', '끌', '런', '개', '놓', '갇', '욱', '관', '잃', '태', '앙', '렇', '란', '봉', '텐', '충', '졸', '끝', '툭', '풍', '냥', '피', '벤', 'ㅡ', '떄', '팡', '지', '컷', '다', '둥', '샤', '컨', '껄', '겉', '안', '흐', '뭔', '렌', '케', '쫙', '프', '빌', '봤', '짖', '과', '자', '직', '릭', '셸']
```
초기에는 subwords 사전이 음절별로 구성되어 있고, 그 길이는 991이다.

### step 2. merge 과정과 subwords 사전 update
```python
# BPE 수행 최대 횟수(병합 횟수)
MAX_MERGES = 10000

for i in progressbar(range(MAX_MERGES)):
    pairs = get_stats(vocab)
    if len(pairs) == 0:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
```
```
 98% (9821 of 10000) |################## | Elapsed Time: 0:01:53 ETA:   0:00:01
```
```python
subwords_dict = vocab_segmentation(vocab)
print('subwords 사전의 개수: {}'.format(len(subwords_dict)))
pprint(subwords_dict[:50])
```
```
subwords 사전의 개수: 5269
['허각씨뿐이었어요</w>',
 '좋았을</w>',
 '끝냄</w>',
 '복수극</w>',
 '그렷습니다</w>',
 'ㅋㅋㅋ애기영화</w>',
 '감정도</w>',
 '꿀잼입니다</w>',
 '결말에서</w>',
 '돌아가신</w>',
 '거북하다</w>',
 '조금이라도</w>',
 '약간의</w>',
 '사실적으로</w>',
 '갑인듯</w>',
 '엉망으로</w>',
 '힘이</w>',
 '찾아가는</w>',
 '품엇던</w>',
 '천하의</w>',
 '인생에</w>',
 '나이라면</w>',
 '아니었다면</w>',
 '찌질이들의</w>',
 '견자단을</w>',
 '슬퍼서</w>',
 '노력했다</w>',
 '쌀밥을</w>',
 '배우</w>',
 '생각된다</w>',
 '나오지</w>',
 '기억함</w>',
 '나온다</w>',
 '여전하구나</w>',
 '재빠르고</w>',
 '싶습니다</w>',
 '왜이렇게욕을하셍휴ㅠㅠ</w>',
 '기대하고</w>',
 '기분이</w>',
 '하나</w>',
 '있자나</w>',
 '공포영화를</w>',
 '누님들</w>',
 '또한</w>',
 '학예회급</w>',
 '주목받아</w>',
 '진정으로</w>',
 '확실한</w>',
 '싶지도</w>',
 '독립영화도</w>']
```
```python
print('ㅋㅋㅋㅋㅋㅋㅋ</w>' in subwords_dict)
```
```
True
```
이전에 했던 품사 기반 전처리 방식과는 달리 'ㅋㅋㅋㅋㅋㅋㅋ'과 같은 단어가 단어 사전에 포함되어 있는 것을 볼 수 있다.

## BPE 알고리즘 with 조사 전처리

앞의 BPE 알고리즘을 수행한 subwords 사전의 결과를 보면 조사들이 계속 포함되어 나타나는 것을 볼 수 있다. 따라서 조사를 미리 제거한 후에 BPE 알고리즘을 적용해보자.
```python
# 조사 리스트
p = re.compile('\n')
with open(r'C:\Users\dpelt\Downloads\KorStems\JosaEomi\josa.txt', 'r') as f:
    temp = f.readlines()
JOSA_list = [p.sub('', word) for word in temp][:-1]
print(JOSA_list[:10])
```

> 한글 조사 목록 출처: [http://nlp.kookmin.ac.kr/data/han-dic.html](http://nlp.kookmin.ac.kr/data/han-dic.html)

```
['가', '같이', '같이나', '같이는', '같이는야', '같이는커녕', '같이도', '같이만', '같인', '고']
```
다음을 조사를 제거하는 전처리 함수이다.
```python
def clean_josa(sent):
    result = deepcopy(sent)
    if type(result) == str:
        for josa in JOSA_list:
            h = re.compile('{}$'.format(josa))
            result = ' '.join(list(map(lambda x: h.sub('', x) if h.search(x) else x, result.split())))
    else:
        result = ''
    return result
```
조사 전처리를 추가한 경우에 앞과 동일하게(parameter 설정이 동일함) BPE 알고리즘을 거친 후의 결과를 확인해보자.
```python
subwords_dict = vocab_segmentation(vocab)
print('subwords 사전의 개수: {}'.format(len(subwords_dict)))
pprint(subwords_dict[:50])
```
```
subwords 사전의 개수: 4427
['극장가서</w>',
 '어릴때</w>',
 '택하</w>',
 '끝냄</w>',
 '한편더찍었으면</w>',
 '복수극</w>',
 '올드함</w>',
 '넘</w>',
 'ㅋㅋㅋ애기영화</w>',
 '정말잘봤습니다눈물</w>',
 '다니겠지</w>',
 '알았네</w>',
 '미소</w>',
 '야동</w>',
 '끌고간게</w>',
 '작품인데</w>',
 '예쁨</w>',
 '산</w>',
 '국수주</w>',
 '보는데</w>',
 '팀</w>',
 '만나기</w>',
 '돌아가신</w>',
 '로맨스</w>',
 '내민</w>',
 '그거</w>',
 '여주인공</w>',
 '쓰레기들</w>',
 '카메</w>',
 '아까웠습니</w>',
 '안</w>',
 '더지니어스</w>',
 '끝내주게</w>',
 '선보인</w>',
 '잔인함</w>',
 '재조명</w>',
 '심한</w>',
 '이어받아서</w>',
 '좋으니까</w>',
 '재난영화냐ㅡㅡ</w>',
 '법은강한자</w>',
 '님</w>',
 '마무리</w>',
 '짰</w>',
 '했</w>',
 '구석</w>',
 '갑인듯</w>',
 '손색없</w>',
 '주제의식</w>',
 '잘만들었</w>']
```





<!--stackedit_data:
eyJoaXN0b3J5IjpbMjAwMDg0NTAzMywyMDEwODcyMTA4XX0=
-->