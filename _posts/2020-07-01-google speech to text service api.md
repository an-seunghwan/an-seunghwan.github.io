---
title: "Google speech to text service api"
excerpt: "Google S2T 활용해보기"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2020-07-01 21:00:00 -0000
categories: 
  - etc
tags:
  - 
---

최근에 KMOOC 사업에 참여하게 되면서, 강의영상의 자막을 제작해야하는 일이 있었다. 하지만 강의영상을 직접 들으면서 일일이 받아 쓰는 것은 큰 무리가 있었고, 다른 좋은 방법을 찾던 중 Google의 speech to text service api를 사용하기로 하였다.

본 포스팅에서는 Google s2t의 service api의 가입 방법은 소개하지 않고, 가입 이후에 활용하는 방법에 대해 중점적으로 소개하고자 한다. api와 관련된 내용이나 확인은 [https://console.cloud.google.com/](https://console.cloud.google.com/)에서 하면 된다.

## 1. api 준비
정식 api가 아니라, google에서 제공하는 service api를 사용하였다. 성공적으로 service api를 받았다면, 다음과 같이 Google Cloud Platform > API 및 서비스 > 사용자 인증 정보 에서 다음과 같은 이메일 계정을 확인할 수 있다. ...@...와 같은 형식으로 써져있다.

<center><img  src="https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/assets/img/s2tapi.jpeg?raw=true" width="400"  height="100"></center>

## 2. 음성파일 준비
.wav 형식의 음성파일을 준비해야한다. 이 때, 


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI0ODE5MTYwMCwtNzQzMzAyODMxLDIwMz
k5OTI5OCwxOTUzMzEyNjczLC0xNzQ5MTAzMTgzXX0=
-->