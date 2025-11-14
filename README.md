# Densepose 基于 Detectron2 (20251113改动)

## 目的
简化densepose的获取难度，直接 image -> api -> densepose_img 删除无关代码

输入 image \
输出 image叠加densepose图 和 黑色背景叠加densepose图

## 环境
- 首先是 安装环境，可以参考 projects/DensePose/requirements.txt （我的环境 py310 torch2.7 cu126）
- 其次 参考 detectron2 官网的安装教程 (主要是要把 detectron2 也 pip install一下，其他环境不一定要一样,比如 torch 2.0 cu118 版本 其他版本适配也行的)

## ckp
下载 https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl 到 projects/DensePose 下

> 参考 博客 https://blog.csdn.net/p13672294967/article/details/140950640

## 新增内容

> 见 projects/DensePose

## 测试and运行

1. 尝试运行 demo_run.py [projects/DensePose/demo_run.py] 如果可行，说明环境基本没问题 (记得放一张图片到这个目录下，不要问傻问题)
2. 运行 flask_densepose.py 开启api服务，关于输入的-c 0 -p 8080 分别表示 cuda port，逻辑在util_for_argparse.py中查看
3. 运行 flask_densepose_query.py 测试api是否可行，输入 -p 与 上述api 接口一致

> 如果flask报错，我经常遇到一个什么escape无法导入and Werkzeug报错 \
> 就执行这个 pip install flask==2.2.2 Werkzeug==2.2.2