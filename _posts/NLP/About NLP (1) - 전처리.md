## tokenizing
* `.`, `,`, `/`, `&` 등의 특수문자 등을 모두 제외하면 안된다.
	- binary classifier를 이용해 `.`의 단어의 일부분(약어로 쓰이는 경우)인 경우와 문장의 구분자인 경우를 구분
	- **binary classifier는 일정한 규칙 또는 머신러닝으로 학습한 모형일 수 있다.**
	- 다른 tokenizing 여부가 쉽게 결정되지 않는 경우들에 대해 적용?
* 단어 내에 띄어쓰기가 있거나, `-`으로 연결되어 있는 경우, 이를 하나의 단어로 인식해야 한다.
* **문장 tokenizing**: how to...?

## 품사 태깅(Part-of-speech tagging)
* 유의미한 단어의 품사만을 활용하면 모형의 성능이 향상될 수 있다.
	* 명사, 동사, 형용사 등
	* 의미있는 품사 단어만을 사용 vs 전체 단어 사용 **(성능 비교)**




<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA5NjgzOTM2MiwtODc3Mzc1MTNdfQ==
-->