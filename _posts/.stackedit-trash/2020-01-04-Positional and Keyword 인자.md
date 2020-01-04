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
## positional argument unpacking
- 위치인자: 함수의 인자를 입력할 때 해당 순서대로 입력을 받는 인자
- list나 tuple과 같이 index가 존재하는 객체를 `*`를 앞에 붙여 함수에 인자로 전달

```python
def my_function(*args):
    return args[1], args[2], args[0]

print(my_function(*[1, 2, 3]))
```
```
(2, 3, 1)
```

## keyword argument unpacking
- 특정한 값으로 지정된 argument
- dictionary type 변수에 `**`를 앞에 붙여 함수에 인자로 전달
- positional argument보다 뒤에 등장해야한다

```python
def my_config(**kwargs):
    print('name : ', kwargs['name'])
    print('learning_rate : ', kwargs['learning_rate'])
    print('split_ratio : ', kwargs['split_ratio'])

config_dict = {'name': 'Adam', 'learning_rate': 1e-3, 'split_ratio': 0.2}
my_config(**config_dict)
```
```
name :  Adam
learning_rate :  0.001
split_ratio :  0.2
```

> [https://wikidocs.net/16048](https://wikidocs.net/16048)를 참고하여 본 게시글을 작성하였음을 밝힙니다!

<!--stackedit_data:
eyJoaXN0b3J5IjpbNTgzMTg2Mzk1XX0=
-->