# 安装uv

```shell
pip install uv

uv --version
```

# 查询python
```shell
# 列出系统中所有可用的 Python 解释器
uv python list
```

# 创建虚拟环境
```shell
# uv venv
# 默认在当前目录创建 .venv

# 指定 Python 版本（需已安装该版本）
# uv venv -p 3.11
# uv venv --python python3.12

# 指定路径
uv venv ~/.convert -p 3.10
```

# 查询卸载包
```shell
uv pip list
uv pip show requests
uv pip uninstall requests
```

# 安装包
```shell
# 激活虚拟环境后（或直接指定）
source ~/.convert/bin/activate  # Linux/macOS
# 安装包
# uv pip install requests flask
# 从 requirements.txt 安装
uv pip install -r model-convert/requirements.txt
```

