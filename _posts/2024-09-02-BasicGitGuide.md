---
title: "[강화시스터즈 2기] 기초 깃허브 사용법"
author: 이지민
tags: [24-2, 개념정리]
pages: papers
style: fill  # fill / border 
color: dark
description: " "
---

이 자료는 동아리 활동 전반에 걸친 과제 제출 및 하반기 협업을 위해 마련되었습니다.  
대략적인 기초 깃/깃허브의 기능을 파악하는 것을 목표로 하며, 단계 별 사용법은 되어 있지 않습니다.  
구체적인 사용법은 **하단의 레퍼런스 링크들 + 카톡방의 자료 + 인터넷 검색**을 이용해 주세요.
{: .notice}

## 버전 관리와 Git
### 버전 관리
버전 관리란, 파일을 주기적으로 저장해 다양한 버전을 만들어두고 사후적으로 특정 시점의 버전에 접속하는 것이다. 
<img src="{{ site.baseurl }}/assets/image/Articles/Git/basic/935803b910910.jpeg" alt="Description of Image" width="400">  


이 익숙한 상황이 바로 버전 관리다. 이 웃픈 상황은 클라이언트와의 소통을 위해 혹은 몇 가지 수정 때문에 여러 가지 버전이 생겨난 상황이지만, 버전 관리는 말단 부가 아닌 프로젝트 관리의 모든 순간에 적용될 수 있다. 이런 방식의 버전 관리는 파일을 복제해 관리하는 방식으로, 어느 부분을 수정했는지가 명확하지 않고 실수로 삭제ㆍ어디를 수정했는지 알 수 없는 문제점이 있다. 이런 여러 문제를 해결해 버전을 관리하는 시스템이 바로 깃이다. 

### Git
#### 1. 분산 버전 관리 시스템(DVCS)
<table border="0">
  <tr>
    <td> 
    중앙 집중식 버전 관리 시스템
    </td>
    <td> 
    분산 버전 관리 시스템
    </td>
  </tr>
 <tr>
    <td> 
    <img src="{{ site.baseurl }}/assets/image/Articles/Git/basic/centralized.png" alt="Description of Image"> 
    </td>
    <td> 
    <img src="{{ site.baseurl }}/assets/image/Articles/Git/basic/distributed.png" alt="Description of Image"> 
    </td>
  </tr>
</table>

여러 버전 관련 시스템 중 깃은 분산 버전 관리 시스템(DVCS)이다. 분산 버전 관리 시스템은 중앙 집중식 버전 관리 시스템의 중앙 서버 과의존성 문제를 해결한 시스템이다. 중앙 집중식 버전 관리 시스템은 중앙 서버에 모든 버전 관리 로그와 파일이 있고 클라이언트가 필요한 버전의 파일을 받아 사용한다. 이 방식은 누가 어떤 작업을 하고 있는지가 명확해 관리가 편하다는 장점이 있지만, 중앙 서버에 모든 정보가 담겨져 있고 관리를 도맡아 하기 때문에 중앙 서버의 문제가 치명적이라는 단점이 있다. 분산 버전 관리 시스템의 클라이언트는 중앙 서버에 있는 모든 로그와 파일을 클론한다. 중앙 집중식 버전 관리 시스템과 달리 모든 클라이언트들에게 원본 파일과 로그가 있어, 중앙 서버가 다운되어도 데이터를 유지할 수 있게 만든다.  

#### 2. 왜 깃을 사용하는가? 
Git은 분산 버전 관리 시스템(DVCS)을 지원하는 유일한 버전 관리 시스템은 아니다. 하지만 깃은 타 버전 관리 시스템과 달리 관리하는 파일들의 버전을 스냅샷처럼 저장한다. 일반적인 분산 버전 관리 시스템(델타 기반 버전 관리 시스템)이 시간 순으로 어떤 파일이 바뀌었는지를 저장한다면, 깃은 파일이 변경된 순간의 모든 파일들의 상태를 묶어서 버전으로 취급한다.

<table border="0">
  <tr>
    <td> 
    일반적인 분산 버전 관리 시스템의 버전 로그
    </td>
    <td> 
    깃의 버전 로그 
    </td>
  </tr>
 <tr>
    <td> 
      <img src="{{ site.baseurl }}/assets/image/Articles/Git/basic/deltas.png" alt="Description of Image"> 
    </td>
    <td> 
      <img src="{{ site.baseurl }}/assets/image/Articles/Git/basic/snapshots.png" alt="Description of Image"> 
    </td>
  </tr>
</table>  

이런 스냅샷 형 버전 저장 방식은 동일한 주제 아래의 코드 변경을 묶어서 변경 사항을 확인할 수 있다는 장점이 있다.   

#### 3. Git의 상태 
Git은 파일을 수정할 때마다 생성되며, Git 데이터베이스에 저장된다.  
수정된 파일은 `CommittedㆍModifiedㆍStaged`라는 3가지 상태로 표현되며, 상태는 깃을 이용한 버전 관리에서 매우 중요한 개념이다.  

1. **Committed** : 데이터가 로컬 데이터베이스에 안전하게 저장됐다는 것을 의미한다.

2. **Modified(Unstaged)** : 수정한 파일을 아직 로컬 데이터베이스에 커밋하지 않은 것을 말한다.

3. **Staged** : 현재 수정한 파일을 곧 커밋할 것이라고 표시한 상태를 의미한다.

:: 가볍게 확인해보기 ::  

소스트리를 기준으로 파일의 상태를 확인해보면 다음과 같다.  

<img src="{{ site.baseurl }}/assets/image/Articles/Git/basic/src.png" alt="Description of Image">  

깃이 관리하는 공간에서 변경된 부분들은 전부 Modified(Unstaged) 상태가 된다.  
이때 커밋을 하고자하는 수정사항들을 선택하면, Staged 상태가 된다.  
이후 수정사항을 기술하고 커밋을 진행하면 변경된 파일은 최종 상태인 Commited 상태가 된다. 


### GitHub
<img src="{{ site.baseurl }}/assets/image/Articles/Git/basic/github-logo.png" alt="Description of Image" width="300">  
깃이 버전 관리와 협업을 가능하게 만들어주는 툴이라면, 깃이 모여드는 깃허브는 협력이 가능한 장소를 제공해준다. 웹 기반 오픈 소스를 기반으로 하며, Git 파일을 온라인으로 접속할 수 있고 협업과 관련된 여러 툴을 제공한다. 


## Repository(저장소)
레파지토리는 모든 코드와 파일, 그리고 모든 파일의 커밋 기록을 갖고 있는 저장소다. 저장소는 내 컴퓨터에 저장되는 로컬 저장소와 깃허브와 같이 웹에 저장되는 원격 저장소로 나뉠 수 있다. 

<img src="{{ site.baseurl }}/assets/image/Articles/Git/basic/Repo.png" alt="Description of Image"> 

위 그림에 있는 용어들은 여러 챕터에 나누어 기술되어 있다. 따라서 글을 따라 읽어 개념을 잡은 뒤 다시 돌아오는 것을 추천한다.
{: .notice--danger}

- **로컬 저장소 (Local Repository)**

  - **Local Repo(Git Direction)** : Git의 프로젝트 데이터가 저장되는 장소며, 가장 핵심적인 부분이다.  
  - **Staging Area** : 커밋하고자 하는 깃 데이터를 선택해두는 장소다.    
  - **Working Directory** : 현재 작업을 하고 있는 공간이며, 이전의 버전을 wd로 만들기 위해서는 local repo에서 버전을 checkout해야 한다. 
   
<br>  

- **원격 저장소 (Remote Repository)**
  - Github와 같이 외부에서 접속할 수 있는 공간에 코드 및 깃이 저장되어 있다. 


### 로컬 저장소 <-> 원격 저장소 
협력을 위해서는 로컬 저장소와 원격 저장소 간의 소통이 중요하다. 로컬로 각기 작업한 내용을 원격 저장소에 합치고, 백업하는 것이 중요하기 때문이다. 저장소와 저장소 사이를 연결하기 위해 사용하는 기능에는 `Tracking, Pull & Push, Clone, Fork`가 있다. 

**1. Tracking(추적)**  
- tracking : 로컬 repo와 원격 repo를 연결한다. 

**2. Pull, Push**  
<img src="{{ site.baseurl }}/assets/image/Articles/Git/basic/remote_local.png" alt="Description of Image" width="200">  

- push : 로컬 repo 의 commit 들을 원격 repo 에 반영한다. 
- pull : 원격 repo 의 commit 들을 로컬 repo 로 반영한다.  

**3. Clone(클론)**
- clone : 내 컴퓨터에 원격 repo의 파일을 복제해 local repo를 만든다. 이때 git direction은 새로 만들어진다. 

**+. Fork(포크)**
- fork : 클론과 같이 원격 repo를 복제하지만, 원본 repo가 갖고 있는 git history를 그대로 가져온다. fork는 원격 repo 사이의 작업이며, fork된 repo는 원본 repo에 기여하거나, 수정된 원본 repo와 수시로 동기화할 수 있다. 

## Git Basic  
### Commit(커밋)
커밋은 스냅샷을 찍어 변경 정보를 저장하는 것이다.  
`누가(author), 언제(시간), commit 시점의 파일 상태, commit 메시지`가 담겨 있다.  
- **add(혹은 staging, 스테이징)** : commit에 반영할지 안할지는 파일 단위로 선택하는 행동 
- **commit history** : commit한 순서대로의 리스트   

커밋은 여러 파일에 걸친 수정 사항을 묶어 진행할 수 있기에, 같은 목적 하에 진행된 수정은 같이 커밋하는 것이 일반적이다. 예를 들어, 여러 env가 각기 다른 py file에 저장되어 있고 공통적인 알고리즘 수정이 있었을 때 'env XX 알고리즘 수정'으로 커밋을 묶어 등록할 수 있다. 

### Pull / Push
[로컬 저장소 <-> 원격 저장소 / Pull & Push](#로컬-저장소---원격-저장소)에 기술되어 있다. 

### Branch
브랜치는 버전 관리의 핵심이다. 브랜치는 일종의 분기점의 역할을 하는데, 이를 통해 독립적인 병렬 작업 공간을 만들 수 있다. 
브랜치는 독립적인 작업 공간을 만들어 주기 때문에 여러 명이 동시에 작업을 할 수 있게 만들어 주며, main 코드를 수정하지 않고 작업을 진행할 수 있기 때문에 문제 해결에 용이하다.  

### Merge & Merge Conflict
 분리된 브랜치는 merge 기능을 통해 합칠 수 있다. 이때 중요한 것은 각 branch가 각기 다른 파일을 수정해 겹치는 파일이 없어야 오류가 나지 않는다는 점이다. 다른 branch에서 동일한 파일이 수정되었을 때 merge conflict가 발생한다. merge conflict가 발생할 때는 아래와 같이 각기 다른 브랜치가 어떤 부분을 어떻게 수정 했는지가 나온다.   

```
<<<< <<< HEAD
{현재 브랜치의 다른 파일 내용}
=======
{충돌나는 브랜치명 또는 commit에서의 다른 파일 내용}
>>> >>>> 충돌나는 브랜치명 또는 commmit 아이디
```
실제로는 <가 전부 이어져 있지만, git은 `<<<<<<< HEAD , ======= , >>>>>>> 충돌나는 브랜치명 또는 commmit 아이디` 양식으로 Merge Conflict를 인식하기 때문에 편집자에게도 본 양식을 작성한 것만으로 merge conflict가 발생해 부득이하게 분리해두었다.
{: .notice}

merge conflict가 발생했을 때는 유지하고 싶은 버전의 코드만을 남기고 `<<<<<<< HEAD , ======= , >>>>>>> 충돌나는 브랜치명 또는 commmit 아이디`을 전부 삭제하면 해결된다. 

### pull request(PR)
pull request는 내 브랜치를 main이나 주요 브랜치에 머지하기 전, 이 브랜치를 병합해도 되는지 확인을 받는 단계다. PR을 받는 사람들은 이 request를 수락할 수도, 거절할 수도 있다. 수락하면 그 즉시 브랜치 사이의 머지가 진행된다. PR은 같은 레파지토리 내에서 진행되기도 하며, fork된 레파지토리와 원본 레파지토리 사이에서도 이루어진다. 

## 협업과 Git(Hub)
깃과 깃허브를 이용해 협업을 하기 위해서는 기본적인 깃 사용법 숙지도 중요하지만, 깃을 어떻게 관리할 것인가를 정하는 것도 중요하다. git branch를 다루는 유명한 전략에는 git flow와 github flow가 있으며, 상황에 따라 다르게 이용하기도 한다. 

### Git Flow/GitHub Flow
[Git Flow / GitHub Flow 정리](https://hudi.blog/git-branch-strategy/)

### issue
GitHub에서 등록할 수 있으며, 구현해야 하는 문제 상황을 공지하거나 일을 나눌 때 사용할 수 있다.  
commit할 때 `#issue_num`을 적으면 해당 이슈에서 커밋 내용을 확인할 수 있다.  

## 레퍼런스
1. [https://git-scm.com/book/ko/v2/시작하기-Git-기초](https://git-scm.com/book/ko/v2/시작하기-Git-기초)
2. [https://github.com/devAon/Eclipse-GitHub-Coraboration-Tutorial?tab=readme-ov-file](https://github.com/devAon/Eclipse-GitHub-Coraboration-Tutorial?tab=readme-ov-file)
3. 
