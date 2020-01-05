## 데이터 
* 네이버 영화 리뷰 데이터 'Naver sentiment movie corpus v1.0'
* 출처: [https://github.com/e9t/nsmc](https://github.com/e9t/nsmc)
* 본 글에서는 평가 데이터만을 예시로 사용한다.

## setup
```python
import pandas as pd
import re
import pickle
from tqdm import tqdm
from konlpy.tag import Okt
```
## 파일 크기 확인
```python
import os

def get_file_size(file_name):
    size = round(os.path.getsize(DATA_PATH + '/ratings_test.txt') / 1000000, 2)
    print('file size: {} MB'.format(size))

FILE_NAME = 'ratings_test.txt'
get_file_size(FILE_NAME)
```
```
file size: 4.89 MB
```
## 파일 읽기
```python
data = pd.read_csv(DATA_PATH + '/' + FILE_NAME, 
                   header=0,
                   delimiter='\t',
                   quoting=3)
data.head()
```
```
        id                                           document  label
0  6270596                                                굳 ㅋ      1
1  9274899                               GDNTOPCLASSINTHECLUB      0
2  8544678             뭐야 이 평점들은.... 나쁘진 않지만 10점 짜리는 더더욱 아니잖아      0
3  6825595                   지루하지는 않은데 완전 막장임... 돈주고 보기에는....      0
4  6723715  3D만 아니었어도 별 다섯 개 줬을텐데.. 왜 3D로 나와서 제 심기를 불편하게 하죠??      0
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTc3NTI0ODMsMTYzMzY5MzI1NSw2OTM5MD
A4OTksLTIwMzg2NzkyODJdfQ==
-->