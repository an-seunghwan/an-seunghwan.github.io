---
title: "Positional and Keyword 인자"
excerpt: Positional and Keyword 인자에 관한 기초 내용
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-04 19:10:00 -0000
categories: 
  - python basic
tags:
  - 
---
## Exception 다루기
```python
import sys
def get_ascii(s):
    '''입력된 문자를 ascii 코드로 변환'''
    try:
        result = ord(s)
    except (ValueError, TypeError) as e: # exception의 정보를 얻기 위해 e로 저장
        print('Error info: ', e, file=sys.stderr) # error의 정보를 출
        raise TypeError("길이 1의 문자가 입력되어야 합니다.") # 특정한 error를 발생
    else:
        print("예외가 발생하지 않았습니다.") # except 부분이 실행되지 않으면 실행
    finally:
        print("반드시 실행되는 부분!")
    return result
```
```python
get_ascii('가')
```
```
예외가 발생하지 않았습니다.
반드시 실행되는 부분!
Out[30]: 44032
```
```python
get_ascii(1)      
```
```
반드시 실행되는 부분!
Error info:  ord() expected string of length 1, but int found
Traceback (most recent call last):

  File "<ipython-input-31-e7656cf8b6ad>", line 1, in <module>
    get_ascii(1)

  File "<ipython-input-23-77f91ee21e54>", line 11, in get_ascii
    raise TypeError("길이 1의 문자가 입력되어야 합니다.")

TypeError: 길이 1의 문자가 입력되어야 합니다.
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTUzODYxMDQzMF19
-->