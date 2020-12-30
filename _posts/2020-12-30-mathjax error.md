---
title: "github.io 수식 오류 해결하기!"
excerpt: "mathjax를 사용할 때 수식이 제대로 보이지 않는 경우의 해결법"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2020-12-30 20:00:00 -0000
categories: 
  - github.io
tags:
  - mathjax
---

이 블로그는 jekyll - minimal mistake 테마를 이용해서 만들었는데, 수식을 입력하기 위해서 그동안 mathjax를 사용해왔다. 

하지만 최근부터 블로그에서 수식이 변환된 형태로 제대로 출력되지 않는 문제가 있어, 이를 해결하는 방법을 찾았고 이를 공유하고자 한다! 잘 이해가 안되시는 분들은 제 블로그의 [github link](https://github.com/an-seunghwan/an-seunghwan.github.io)에 들어가서 직접 확인해보시는 것도 좋을 것 같습니다 :)

## 1. _config.yml 확인하기

_config.yml 파일을 확인했을 때, 아래와 같이 markdown engine이 kramdown으로 되어있어야 한다!

```yml
# Conversion
markdown: kramdown
highlighter: rouge
lsi: false
excerpt_separator: "\n\n"
incremental: false

# Markdown Processing
kramdown:
input: GFM
hard_wrap: false
auto_ids: true
footnote_nr: 1
entity_output: as_char
toc_levels: 1..6
smart_quotes: lsquo,rsquo,ldquo,rdquo
enable_coderay: false
```

## 2. _includes/scripts.html 파일 수정하기

_includes/scripts.html
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgzOTExNjc0OSwtMzk5MzU5NjYzXX0=
-->