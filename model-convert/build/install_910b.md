# 华为910B平台安装部署手册

本文档旨在指导开发者在华为910B平台上部署和运行模型转换与推理服务。文档涵盖了系统环境依赖、代码部署说明以及相关配置的详细信息。

## 1. 系统环境依赖

### 1.1 硬件环境
- 华为昇腾910B NPU设备

### 1.2 软件环境
#### 基础系统依赖
- Ubuntu 22.04 LTS
- Python 3.10.0
- cmake >= 3.5.1
- make
- gcc >= 4.8.5
- g++ >= 4.8.5
- unzip
- zlib-devel (zlib1g-dev)
- libffi-devel (libffi-dev)
- openssl-devel
- pciutils
- net-tools
- gdbm-devel (libgdbm-dev)

#### CANN工具链
- CANN版本：8.0.0
- 安装路径：`/usr/local/Ascend/ascend-toolkit/8.0.0`

#### Python依赖包
##### 第一步依赖 (requirements-hw-step1.txt)
```
decorator==4.4.0
sympy==1.5.1
cffi==2.0.0
attrs==25.3.0
pyyaml
pathlib2
scipy
requests==2.32.3
psutil==7.0.0
absl-py==2.1.0
opencv-python==4.12.0.88
ultralytics==8.3.156
onnx==1.17.0
onnxruntime==1.16.0
fastapi==0.119.0
uvicorn==0.38.0
minio==7.2.18
torch==2.8.0
torch_npu==2.8.0
torchvision==0.23.0
```

##### 第二步依赖 (requirements-hw-step2.txt)
```
numpy==1.24.0
```

#### 服务依赖
- Supervisor (用于进程管理)
- MinIO (用于文件存储)

## 2. 代码部署说明

### 2.1 项目结构
整个项目包含前后端分离的架构，其中华为910B平台相关的代码位于`model-convert`目录下：

```
model-manager/
├── model-convert/            # 模型转换与推理服务
│   ├── config/               # 配置文件目录
│   │   └── config.yaml       # 平台配置文件
│   ├── rest_simple/          # 华为910B平台推理和模型转换服务主文件
│   │   └── rest_predict_910b.py
│   ├── supervisor/           # Supervisor配置文件目录
│   │   ├── supervisord.conf  # Supervisor主配置文件
│   │   └── SRestSimple       # 华为910B服务的Supervisor配置文件
│   ├── convert/              # 模型转换脚本目录
│   │   └── onnx_to_om.py     # ONNX到OM模型转换脚本
│   ├── predict/              # 推理脚本目录
│   │   └── predict_om.py     # OM模型推理脚本
│   ├── common/               # 公共模块
│   ├── service/              # 任务管理服务
│   ├── tools/                # 工具模块
│   │   └── handle_file_minio.py  # MinIO文件处理工具
│   └── build/
│       └── install_910b.md   # 本安装部署手册
```

### 2.2 部署步骤

#### 2.2.1 环境准备（可选，一般910B环境可能已经支持）
```bash
# 安装系统依赖
sudo apt update
sudo apt install -y python3.10  unzip zlib1g-dev libffi-dev libssl-dev pciutils net-tools libgdbm-dev
# sudo apt install -y cmake make gcc g++
# 安装CANN 8.0.0
# 请参考华为官方文档进行安装: https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0033.html
```

#### 2.2.2 Python虚拟环境配置
```bash
# 创建虚拟环境
cd /path/to/model-manager
python3.10 -m venv venv_910b
source venv_910b/bin/activate

# 安装第一步依赖
pip install -r model-convert/build/build_asend/requirements-hw-step1.txt

# 安装第二步依赖
pip install -r model-convert/build/build_asend/requirements-hw-step2.txt
```

#### 2.2.3 配置文件设置
华为910B平台的相关配置位于`model-convert/config/config.yaml`文件中，主要包括：
- MinIO服务器配置（地址、访问密钥等）
- MySQL数据库配置（如果使用）
- 华为昇腾设备配置（SoC版本等）

请根据实际环境修改配置文件中的相应参数。

#### 2.2.4 配置环境变量
在`rest_simple/rest_predict_910b.py`中已经设置了必要的环境变量：
```python
os.environ["ASCEND_HOME"] = "/usr/local/Ascend/ascend-toolkit/8.0.0"
os.environ["LD_LIBRARY_PATH"] = (
    f"{os.environ['ASCEND_HOME']}/lib64:"
    "/usr/local/Ascend/driver/lib64:"
    f"{os.environ.get('LD_LIBRARY_PATH', '')}"
)
os.environ["PATH"] = f"{os.environ['ASCEND_HOME']}/bin:{os.environ.get('PATH', '')}"
os.environ["PYTHONPATH"] = (
    f"{os.environ['ASCEND_HOME']}/python/site-packages:"
    f"{os.environ.get('PYTHONPATH', '')}"
)
```

#### 2.2.5 Supervisor配置

在`model-convert/supervisor`目录下，有多个Supervisor配置文件：

- `SRestSimple`: 推理服务进程管理配置
- `Sconvert`: 模型转换服务进程管理配置（该版本忽略）
- `Sfrpc`: FRPC内网穿透服务进程管理配置（需要内网穿透才关注）

此外，还有一个主配置文件`supervisord.conf`，它包含了Supervisor的核心配置。

##### 2.2.5.1 supervisord.conf配置详解

`supervisord.conf`是Supervisor的主配置文件，主要包含以下配置项：

1. **Unix HTTP Server配置**：
   - `file`: 定义socket文件路径，用于supervisorctl与supervisord通信

2. **Supervisord全局配置**：
   - `environment`: 环境变量设置，这里设置了PYTHONPATH以包含项目路径, 用于加载模型转换和推理服务


3. **Supervisorctl配置**：
   - `serverurl`: supervisorctl连接supervisord的URL

4. **包含配置**：
   - `files`: 包含其他配置文件的模式，这里设置为`S*`表示包含所有以S开头的配置文件

##### 2.2.5.2 部署Supervisor配置

将`supervisord.conf`配置文件复制到`/etc/supervisor/conf.d/`目录下：

```bash
sudo cp model-convert/supervisor/supervisord.conf /etc/supervisor/conf.d/
```

##### 2.2.5.3 注册Supervisor服务(docker/WSL等环境不支持，直接运行supervisor启动)

为了确保Supervisor在系统启动时自动运行，我们需要将其注册为系统服务。项目提供了`super.service`配置文件，可以用于注册Supervisor服务。
该配置文件中的配置文件supervisord.conf的路径需要根据实际部署环境进行调整。

1. 将服务配置文件复制到系统目录：

```bash
cd model-convert/supervisor/
sudo cp super.service /etc/systemd/system/910b.service
```

2. 重新加载systemd配置：

```bash
sudo systemctl daemon-reload
```

3. 启用并启动Supervisor服务：

```bash
sudo systemctl enable 910b
sudo systemctl start 910b
```

4. 检查服务状态：

```bash
sudo systemctl status 910b
```

注意：`super.service`文件中的路径可能需要根据实际部署环境进行调整。默认配置使用`/root/miniconda3/convert/supervisor/supervisord.conf`作为配置文件路径，如果您的部署路径不同，请相应修改该文件中的路径。

#### 2.2.6 启动服务（手工启动，如果注册Supervisor服务失败）
```bash
cd model-convert/supervisor/
# 启动Supervisor
sudo supervisord -c supervisord.conf

# 管理服务
sudo supervisorctl -c supervisord.conf status     # 查看状态
sudo supervisorctl -c supervisord.conf start rest  # 启动服务
sudo supervisorctl -c supervisord.conf stop rest   # 停止服务
sudo supervisorctl -c supervisord.conf restart rest # 重启服务
```

### 2.3 接口服务说明

华为910B平台的接口服务由`model-convert/rest_simple/rest_predict_910b.py`提供，通过Supervisor进行管理。服务启动后将在默认端口提供RESTful API接口。

#### 2.3.1 主要功能
1. 模型转换任务：将ONNX模型转换为OM格式
2. 图片推理任务：在华为910B设备上执行模型推理
3. MinIO集成：支持从MinIO下载模型和上传推理结果
4. 回调机制：任务完成后通知调用方

#### 2.3.2 配置文件
服务使用`model-convert/config/config.yaml`作为主要配置文件，包含MinIO、MySQL、服务器和华为昇腾等相关配置。

#### 2.3.3 API接口
服务提供以下主要API接口：
1. `/convert` - 模型转换接口
2. `/predict` - 模型推理接口
3. `/callback` - 回调通知接口

具体接口文档可通过访问服务的`/docs`路径查看。

## 3. 验证部署

启动服务后，可以通过以下方式验证部署是否成功：

1. 检查Supervisor状态：
```bash
sudo supervisorctl -c model-convert/supervisor/supervisord.conf status
```

2. 访问API文档：
在浏览器中打开 `http://<服务器IP>:39000/docs` 查看API文档

3. 运行测试脚本：
```bash
cd model-convert
python -m test.test_rest_predict_910b
```

## 4. 其他说明

本部署手册仅针对华为910B平台的模型转换与推理服务。完整的系统还包括前端界面和后端管理系统，它们有独立的部署方式。如需部署完整系统，请参考项目根目录下的相关文档。