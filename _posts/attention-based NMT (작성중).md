> 이 포스팅은 Sequence to Sequence Model(seq2seq), RNN, LSTM, GRU, teacher forcing 등과 관련한 내용을 다루는 [seq2seq 톺아보기 시리즈](https://an-seunghwan.github.io/seq2seq-top-a-bogi/)의 (?)편입니다.

>  본 포스팅은 [Effective approaches to attention-based neural machine translation](https://arxiv.org/pdf/1508.04025.pdf)에 대한 간단한 요약과 번역을 작성한 글입니다. 정확한 내용은 반드시 원문을 참조해 주시기 바랍니다.

## 0. 용어의 번역

- source : 원문 (NMT 모형의 번역 대상)
- attend : 고려되다.

## 2. Neural Machine Translation(NMT)

NMT system은 원문 문장($$)

논문에서는 다음과 같은 두 가지 attention-based 모형을 제안한다: *global* approach에서는 모든 원문의 단어들이 고려된다. 그리고 *local* approach에서는 원문의 단어 중 일부만이 고려된다.

## 논문
LUONG, Minh-Thang; PHAM, Hieu; MANNING, Christopher D. Effective approaches to attention-based neural machine translation. _arXiv preprint arXiv:1508.04025_, 2015.

<!--stackedit_data:
eyJoaXN0b3J5IjpbMjIxNDA3NTA4XX0=
-->