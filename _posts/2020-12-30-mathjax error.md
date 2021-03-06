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

`_config.yml` 파일을 확인했을 때, 아래와 같이 markdown engine이 `kramdown`으로 되어있어야 한다!

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

## 2. _includes/mathjax_support.html 파일 추가하기

`_includes` 폴더에 아래와 같은 내용이 적힌 `mathjax_support.html` 파일을 추가해야한다.

```html
<script type="text/x-mathjax-config">
	MathJax.Hub.Config({
	TeX: {
		equationNumbers: {
		autoNumber: "AMS"
		}
	},
	tex2jax: {
		inlineMath: [ ['$', '$'] ],
		displayMath: [ ['$$', '$$'] ],
		processEscapes: true,
		}
	});
	MathJax.Hub.Register.MessageHook("Math Processing Error",function (message) {
		alert("Math Processing Error: "+message[1]);
	});
	MathJax.Hub.Register.MessageHook("TeX Jax - parse error",function (message) {
		alert("Math Processing Error: "+message[1]);
	});
</script>

<script type="text/javascript" async
	src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
```

(참고: [https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/_includes/mathjax_support.html](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/_includes/mathjax_support.html))

## 3. _layouts/defaut.html 파일 확인하기

_layouts/defaut.html 파일을 확인했을 때, `<head>`와 `</head>` 사이에 아래와 같은 `use_math`와 `mathjax_support.html`에 관한 명령문이 적혀있어야 한다 (코드 첨부가 안되서 아래 제 github link를 참고해 주세요).

(참고: [https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/_layouts/default.html](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/_layouts/default.html))

## 4. _includes/scripts.html 파일 수정하기

가장 중요한 부분인데, 최근까지 잘 되던 수식 입력이 안되는 이유는 이 `scripts.html`파일 때문이었다!!!

`_includes` 폴더의 `scripts.html` 파일에 아래와 같은 코드를 맨 아래에 그대로 추가하면 이제 모든 문제가 해결된다.

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

(참고: [https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/_includes/scripts.html](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/_includes/scripts.html))

## 5. 추가 - 수식 작성법

### 1. inline 작성

수식을 inline으로 작성할 때는 `$a^2 + b^2 = c^2$` 처럼 적으면 $a^2 + b^2 = c^2$ 이렇게 문장 내에서 써진다.

### 2. outline 작성

수식을 outline으로 작성할 때는 

`$$
a^2 + b^2 = c^2
$$` 

처럼 적으면 

$$
a^2 + b^2 = c^2
$$ 

이렇게 별도의 line에서 써진다.

### 3. outline 수식 정렬

outline 수식을 정렬하고 싶을 때는 
```
$$
\begin{aligned} 
a^2 + b^2 &= c^2 \\ 
E &= M \cdot C^2 \\ 
&= xy + \mathbb{E} 
\end{aligned}
$$
```
처럼 적으면 

$$
\begin{aligned} 
a^2 + b^2 &= c^2 \\ 
E &= M \cdot C^2 \\ 
&= xy + \mathbb{E} 
\end{aligned}
$$

이렇게 정렬되어 수식이 써진다.
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA1Njk4NTEzLC00NTA5MzY3OSw5MDk4MT
MzOTEsMTMzNzE5NzcxOCwtMzk5MzU5NjYzXX0=
-->