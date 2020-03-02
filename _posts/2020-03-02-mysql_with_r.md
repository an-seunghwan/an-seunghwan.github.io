---
title: "MySQL을 R에서 사용해보자!"
excerpt: "MySQL 활용하기"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2020-03-02 21:00:00 -0000
categories: 
  - advanced
tags:
  - database
  - mysql
---
## 0. 환경
- server: ubuntu 18.04

## 1. MySQL 설치
```
sudo apt-get update
sudo apt-get install mysql-server
sudo mysql_secure_installation
```

MySQL 접속
```
sudo mysql
```
`sudo`로 접근해야 localhost connection에 관한 error가 발생하지 않음.

## 2. database 생성 및 table 생성
- database 생성
```
create database (dbname);
show databases;
```
- csv file import
- 

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3Mjc5NTI5MjZdfQ==
-->