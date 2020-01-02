---
title: "python module"
excerpt: python module에 관한 기초 내용
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-02 21:30:00 -0400
categories: 
  - python basic
tags:
  - 
---
## python module

### module: script vs import
다음과 같은 python 코드를 생각하자.
```python
def words_to_ascii():
    with open(r'C:\Users\dpelt\Documents\GitHub\an-seunghwan.github.io\assets\etc\sample.txt', 'r', encoding='utf-8') as text:
        lines = text.readlines()
        for line in lines:
            ascii_code = [str(ord(word)) for word in line]
            print(''.join(ascii_code))
            
print(__name__)
if __name__ == '__main__':
    words_to_ascii()
```
1. REPL로 실행하는 경우
![](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/assets/img/ascii1.png?raw=true)

`import words`를 실행하게 되면 module의 이름인 'ascii_encoding'가 출력되는 것을 볼 수 있다.

2. script로 실행하는 경우
![](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/assets/img/ascii2.png?raw=true)

script 형태로 실행하게 되면 이름이 `__main__`으로 출력되고 따라서 함수가 바로 실행되는 것을 볼 수 있다.

### example
다음의 예제는 입력받은 문자열 각 글자별로 ascii 암호화 해주는 module이다.
```python
import sys

def words_to_ascii(text):
    result = []
    tokenized = [x for x in text]
    for word in tokenized:
        result.append(''.join([str(ord(x)) for x in word]))
    return result

def print_ascii(result):
    for code in result:
        print(code)
        
def main(text):
    result = words_to_ascii(text)
    print_ascii(result)
    
print(__name__)
if __name__ == '__main__':
    main(sys.argv[1:]) # 콘솔창에서 sys.argv[0]은 module의 파일명으로 받고 그 뒤 인자들은 함수의 입력값으로써 받는다.
```
![](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/assets/img/ascii3.png?raw=true)

### Docstring
다음과 같이 Docstring을 추가했습니다.
* docstring은 module, 함수, class 정의의 내부 바로 아래 줄에 큰 따옴표 또는 작은 따옴표 3개로 작성한 문자열
* 해당 객체의 **doc** 특수 속성으로 변환됨

1. `help()`를 이용하면 module 전체의 docstring 확인 가능
![](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/assets/img/ascii4.png?raw=true)
2. `help()`를 이용해 각 함수의 docstring과 `__doc__` 속성을 이용해 docstring을 확인
![](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/assets/img/ascii5.png?raw=true)

> [https://wikidocs.net/16048](https://wikidocs.net/16048)를 참고하여 본 게시글을 작성하였음을 밝힙니다!


<!--stackedit_data:
eyJoaXN0b3J5IjpbMTI1NTY4MTU2N119
-->