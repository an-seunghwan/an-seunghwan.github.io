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

## 2. 한글을 위한 utf-8 encoding 설정
(너무 복잡함...)

## 3. database 생성 
- database 생성
```
create database (dbname);
show databases;
```
## 4. table 생성(csv file import)

- sample이라는 table format 지정
```
CREATE TABLE sample(
	id INT NOT NULL AUTO_INCREMENT,
	a VARCHAR(255) NOT NULL,
	b INT NOT NULL,
	c INT NOT NULL,
	d INT NOT NULL,
	e INT NOT NULL,
	f INT NOT NULL,
	g INT NOT NULL,
	h INT NOT NULL,
	i INT NOT NULL,
	j INT NOT NULL,
	k INT NOT NULL,
	l INT NOT NULL,
	m INT NOT NULL,
	n INT NOT NULL,
	o INT NOT NULL,
	p INT NOT NULL,
	q INT NOT NULL,
	r INT NOT NULL,
	s INT NOT NULL,
	PRIMARY KEY (id)
	)
	default character set utf8 collate utf8_general_ci;
```
- sample table format에 맞는 데이터 입력
	- 이때 csv 파일을 utf
```
LOAD DATA LOCAL INFILE '/home/jeon/Desktop/sql_data/report1.csv'
INTO TABLE sample
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s);
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbMjEzMDIxNTg0M119
-->