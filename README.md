# Pose-Compare

# 安装

## 依赖
- Python 3.6+
- [NVIDI CUDA 10.2](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA CUDA DNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)
  
## 安装
```sh
$ pip install mxnet-cu102 -f https://dist.mxnet.io/python/all
$ pip install -r requirements.txt
```

# 运行

提取示例视频数据

```sh
$ python data.py --input 示例视频 --output 示例数据.tsv
```

动作对比
```sh
$ python run.py --input 输入视频 --demo 示例视频 --data 示例数据.tsv