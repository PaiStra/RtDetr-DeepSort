---
title: README
date: 2024-07-10 16:39:45
author: 派小星
---

# README



## Quick start

1.创建环境

```
conda env create -f requirements.yml 	//默认环境名为dt
或
conda env create -f requirements.yml -n NAME	//修改NAME来自行指定名称
```

2.使用脚本

```
python WorldVideoMoreByOnce.py
```



## 使用GPU加速

1.本地配置CUDA和CUDNN

2.在项目中根据本地CUDA版本使用pip下载PyTorch

```
https://pytorch.org/get-started/locally/				//PyTorch官网
```

