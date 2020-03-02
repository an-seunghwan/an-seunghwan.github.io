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
- database(이름은 test) 생성
```
create database test;
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
	- 이때 csv 파일을 '쉼표로 구분된 utf-8 인코딩' 형식으로 저장
```
LOAD DATA LOCAL INFILE '/home/jeon/Desktop/sql_data/report1.csv'
INTO TABLE sample
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s);
```
- table 확인
```
select * from sample;
```

## 5. 사용자 계정 설정
```
grant select on test.* 'user'@'%' by 'userpassword';
```
-  select 권한만을 부여
- `test.*`: test database에 속해있는 모든 파일에 대해 권한 부여
- `'user'@'%'`: 사용자 계정의 id는 user이고 특정 ip 주소를 특정하게 지정하지 않음(모두 가능)
- `'userpassword'`: mysql 비밀번호 규칙에 따른 비밀번호(일반적으로 8자리 이상, 대소문자 1회, 숫자 및 특수문자 1회 이상)
- 따옴표는 반드시 적어줄 것

## 6. 방화벽 설정
- port 확인
```
cd /etc/mysql/mysql.conf.d
sudo nano mysqld.cnf
```
`port=3306`을 확인할 수 있다(일반적으로 3306).

- 방화벽 해제
```
sudo ufw allow 3306/tcp
sudo ufw status
```
```
상태: 활성

목적                         동작          출발
--                         --          --
Apache                     ALLOW       Anywhere
22/tcp                     ALLOW       Anywhere
3306/tcp                   ALLOW       Anywhere
Apache (v6)                ALLOW       Anywhere (v6)
22/tcp (v6)                ALLOW       Anywhere (v6)
3306/tcp (v6)              ALLOW       Anywhere (v6)
```
- bind-address 설정
```
cd etc/mysql/mysql.conf.d  
sudo nano mysqld.cnf  
```
`bind-address`로 시작하는 부분을 주석 처리

- 재시작
```
sudo service mysql restart
```

## 7. R에서 접속하기
```r
library(RMySQL)

```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3MzE3MDg3NzZdfQ==
-->