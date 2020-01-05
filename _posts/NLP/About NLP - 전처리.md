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

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTYzMzY5MzI1NSw2OTM5MDA4OTksLTIwMz
g2NzkyODJdfQ==
-->