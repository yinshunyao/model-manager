# 华为设备
运行环境安装 cann-nnrt 即可， 如果要使用 ACL 工具，需要安装整个tookit

1. 非昇腾设备安装 
系统依赖软件确定
```shell
# 参考 https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0033.html
# 环境要求 Ubuntu 22.04
# 版本约束，确认已安装软件依赖 dpkg -l | grep make
# 如果缺少依赖需要安装 apt-get install cmake
Python==3.10.0
cmake >=3.5.1
make
# 离线推理
gcc>=4.8.5
gcc++>=4.8.5 # g++ --version 检查版本
unzip
zlib-devel # sudo apt install  zlib1g-dev
libffi-devel  # sudo apt install libffi-dev
openssl-devel 
pciutils
net-tools
gdbm-devel  # sudo apt install libgdbm-dev
```

python 依赖安装
```shell
cd envs
../bin/python3 -m venv hw
hw/bin/python3 -m pip install -r requirements-hw.txt
```

cann
```shell
# 下载地址 https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/264595017?idAbsPath=fixnode01|23710424|251366513|22892968|252309113|251168373
cann==8.0.0

# 查询 cann 版本
npu-smi info
# cann 安装情况
ll /usr/local/Ascend/ascend-toolkit/latest/

# 租赁平台
ssh -p 31355 root@connect.gda1.seetacloud.com
PUJlQB1jPqZe

# 如果 cann 找不到，修复
# 替换为实际版本目录（从符号链接可知是8.0.0）
export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/8.0.0
# 更新依赖路径
export PATH=$ASCEND_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/acllib/lib64:$LD_LIBRARY_PATH
source ~/.bashrc 

# 验证库是否可找到
ldconfig -p | grep libascendcl.so 

# 修复权限
sudo chmod -R 755 /usr/local/Ascend/ascend-toolkit/8.0.0


```

2. 昇腾设备安装


# 转换脚本
```shell
atc --model=/root/miniconda3/convert/yolo11n.onnx --framework=5 --output=/root/miniconda3/convert/yolo.om --input_shape=images:1,3,640,640 --soc_version=Ascend910B --precision_mode=allow_fp32_to_fp16 --log=error

atc --model=/root/miniconda3/convert/yolo11n.onnx --framework=5 --output=/tmp/tmp.yolo --input_shape=images:1,3,640,640 --soc_version=Ascend910B  --log=error --output_type=FP32

```


# tookit安装和使用
https://www.hiascend.com/developer
```shell
# om 报错，打印相关信息
amct --model=yolo11n.om.om --print_info
```