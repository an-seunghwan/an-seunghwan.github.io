---
title: "Exception 다루기"
excerpt: Exception에 관한 기초 내용
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-04 19:40:00 -0000
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
        print('Error info: ', e, file=sys.stderr) # error의 정보를 출력
        raise TypeError("길이 1의 문자가 입력되어야 합니다.") # 특정한 error를 발생
    else:
        print("예외가 발생하지 않았습니다.") # except 부분이 실행되지 않으면 실행
    finally:
        print("반드시 실행되는 부분!")
    return result
```
```python
print(get_ascii('가'))
```
```
예외가 발생하지 않았습니다.
반드시 실행되는 부분!
44032
```
```python
print(get_ascii(1))   
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
이때 `Error info:  ord() expected string of length 1, but int found`부분이 `print('Error info: ', e, file=sys.stderr)`에 의해서 출력되는 부분이다. 그리고 `raise TypeError("길이 1의 문자가 입력되어야 합니다.")`부분에 의해 결과물의 `TypeError: 길이 1의 문자가 입력되어야 합니다.`이 출력된다.
```python
print(get_ascii('가나'))
```
```
반드시 실행되는 부분!
Error info:  ord() expected a character, but string of length 2 found
Traceback (most recent call last):

  File "<ipython-input-41-46070fd5061f>", line 1, in <module>
    print(get_ascii('가나'))

  File "<ipython-input-40-4fa272f8b74b>", line 11, in get_ascii
    raise TypeError("길이 1의 문자가 입력되어야 합니다.") # 특정한 error를 발생

TypeError: 길이 1의 문자가 입력되어야 합니다.
```
> 출처: [https://wikidocs.net/16048](https://wikidocs.net/16048)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0NjE5MzQ1NzAsODgzMjY0ODQzLC0xNz
g1NjA0Njc0LDE4MTE0MDE2MjcsODU3NjUyMzRdfQ==
-->