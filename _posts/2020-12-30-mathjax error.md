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

(참고: [https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/_config.yml](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/_config.yml))

## 2. 

(참고: [https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/_includes/mathjax_support.html](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/_includes/mathjax_support.html))

## 3. _layouts/defaut.html 파일 확인하기

_layouts/defaut.html 파일을 확인했을 때, 아래와 같은 `use_math`에 관한 명령문이 적혀있어야 한다.

```html
<head>
	{% include head.html %}
	{% include head/custom.html %}
	{% if page.use_math %}
		{% include mathjax_support.html %}
	{% endif %}
</head>
```

(참고: [https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/_layouts/default.html](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/_layouts/default.html))

## 3. _includes/scripts.html 파일 수정하기

_includes/scripts.html

```html
<script type="text/javascript" async
	src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML">
</script>

<script type="text/x-mathjax-config">
	MathJax.Hub.Config({
	extensions: ["tex2jax.js"],
	jax: ["input/TeX", "output/HTML-CSS"],
	tex2jax: {
		inlineMath: [ ['$','$'], ["\\(","\\)"] ],
		displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
		processEscapes: true
	},
	"HTML-CSS": { availableFonts: ["TeX"] }
});
</script>
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTk4MTY2MzQ1LC0zOTkzNTk2NjNdfQ==
-->