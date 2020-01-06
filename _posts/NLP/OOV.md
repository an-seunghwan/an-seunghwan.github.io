## OOV problem - 단어 분리하기(Byte Pair Encoding, BPE)

**모르는 단어가 등장하여 문제를 해결하기 어려운 상황 = OOV**
따라서 **단어 분리 (subword segmentation)** 를 하여 단어를 여러 단어로 분리해서 이해를 해보려는 시도를 한다.

### 1. BPE
* 데이터 압축 알고리즘
* 예제
```
aaabdaaabac
```
연속적으로 가장 많이 등장한 글자의 쌍 = 'aa' = byte pair
따라서 
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgwODMzMjYyXX0=
-->