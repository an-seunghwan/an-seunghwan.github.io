# blog contents list!

### github.io 블로그 만드는 방법

### docker?

### OOV
* Okt를 적용한 뒤에 BPE를 적용한다면?
* 그럼 BPE의 tokenizer로써의 역할이 무색?

### Keras 번역 시리즈
* [https://www.tensorflow.org/guide](https://www.tensorflow.org/guide) 
* 번역 X : 번역 시리즈 추가
* 번역 O : 공부 시리즈 추가
* masking and padding 내용 추가

### Keras 공부 시리즈
* serialization
	- keras.Model.get_config()
	- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer?version=stable#from_config
* Functional vs subclassing
	- https://medium.com/tensorflow/what-are-symbolic-and-imperative-apis-in-tensorflow-2-0-dfccecb01021
* keras.losses.Reduction
* shape=(-1, 1)
* tf.cast
* https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer?version=stable#dtypes_and_casting_2
* [https://www.tensorflow.org/tensorboard/get_started](https://www.tensorflow.org/tensorboard/get_started)
* 

### Keras visualization
1. 오류 발생...

### 데코레이터
1. 데코레이터 정의 
2. [https://www.tensorflow.org/guide/function](https://www.tensorflow.org/guide/function)

### python class
1. `__init__`, `build`, `call`의 의미+++
2. Keras에서 model, layer class에서 다른점+++
3. Keras class 위주로 어떤 method가 어떻게 작동되는지 확인+++
4. 공부 후에 nested RNN 번역 내용 추가!

### python basic
1. 파이썬 - 기본을 갈고 닦자

### 딥 러닝을 이용한 자연어 처리 입문
1. 형태소
2. 우리나라 말에서 의미를 갖는 단어 품사만을 활용하면 성능 향상?
	* 명사, 동사, 형용사 등
	* 의미있는 품사 단어만을 사용 vs 전체 단어 사용 (성능 비교)

### Keras Details 와 용어들
1. Keras `dtype`의 default는 `float32`
	* error message 확인
	* `numpy.ndarray.astype('float32')`를 이용해 numpy의 dtype 변경하면 오류 해결 가능
	* `float64`를 사용하고 싶다면 `tf.keras.backend.set_floatx('float64')`

### Keras RNN shape & 용어들
* timesteps, units 등
* 공부할 때 그린 그림을 첨부해서 설명하면 더 좋을듯(아이패드 ?)
* instance ?

### LSTM 
* hidden state은 2개?

### Bidirectional RNN

### GRU 

### Feature extraction과 extracting node?

### 분산형 학습 in Keras & MPI

### tokenizing
* `.`, `,`, `/`, `&` 등의 특수문자 등을 모두 제외하면 안된다.
	- binary classifier를 이용해 `.`의 단어의 일부분(약어로 쓰이는 경우)인 경우와 문장의 구분자인 경우를 구분
	- **binary classifier는 일정한 규칙 또는 머신러닝으로 학습한 모형일 수 있다.**

### 품사 태깅
* 의미있는 품사 단어만을 사용 vs 전체 단어 사용 **(성능 비교)**

### 형태소






<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2NzMxNjI4NjIsLTY2MzkxNTg4MCwxNT
U5MzgxMDgyLC0yMTMwODc2NDk0LDQyNTYyMzU1XX0=
-->