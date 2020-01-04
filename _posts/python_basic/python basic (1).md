### False 값
```python
False_values = [0, 0.0, "", [], {}]
for x in False_values:
    print(bool(x))
```
```
False
False
False
False
False
```
이 외에는 모두 `True` 값을 가짐.

### 따옴표
* 큰 따옴표와 작은 따옴표는 혼용 불가
* comma가 없으면 자동으로 합쳐진다.
```python
'abc' 'def'
```
```
'abcdef'
```

### 리스트
* 합치기
```python
[1, 2, 3,] + [4, 5,]
```
```
[1, 2, 3, 4, 5]
```
* `-`는 역순으로 인덱싱
* list copy
```python
l = [1, 2, 3,]
new_l = l[:]
print(new_l == l) # 값은 동일
print(new_l is l) # 다른 변수
```
```
True
False
```
* step 사용
```python
l = [1,2,3,4,5,6,]
print(l[::2])
print(l[::-1]) # 역순 정렬
```
```
[1, 3, 5]
[6, 5, 4, 3, 2, 1]
```
* **list를 반복하는 경우 이는 얕은 복사(shallow copy)**
* 기타 method
```python
l = ['a','a','b','c',]
print(l.index('a'))
print(l.index('b'))
print(l.count('a'))
```
```
0
2
2
```
```python
a = [1, 2, 3]
b = [4, 5, 6]
a.insert(1, 10)
print(a)
a.append(b)
print(a)
a.extend(b)
print(a)
del a[1]
print(a)
del a[a.index(1)] # a.remove(1) 과 동일
print(a)
a.remove(7)
```
```
[1, 10, 2, 3]
[1, 10, 2, 3, [4, 5, 6]]
[1, 10, 2, 3, [4, 5, 6], 4, 5, 6]
[1, 2, 3, [4, 5, 6], 4, 5, 6]
[2, 3, [4, 5, 6], 4, 5, 6]
Traceback (most recent call last):

  File "<ipython-input-151-0b307e062a0f>", line 13, in <module>
    a.remove(7)

ValueError: list.remove(x): x not in list
```
> List(리스트)(4) - 리스트 원소 추가, 삭제 
> To be continued...
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTI2MjQzNDAyNV19
-->