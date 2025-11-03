# docker 安装（可选）
```shell
# 安装Docker
sudo apt update
sudo apt install -y docker.io

# 启动Docker服务并设置开机自启
sudo systemctl start docker
sudo systemctl enable docker

# 赋予当前用户Docker权限（避免每次用sudo）
sudo usermod -aG docker $USER
newgrp docker  # 生效权限
```

# dokcer 镜像地址配置 （可选）
```shell
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://registry.cn-hangzhou.aliyuncs.com"]
}
EOF
# https://registry.cn-hangzhou.aliyuncs.com
# 
registry.cn-beijing.aliyuncs.com
registry.cn-shanghai.aliyuncs.com
registry.cn-shenzhen.aliyuncs.com
sudo systemctl daemon-reload
sudo systemctl restart docker
```
# dokcer 镜像地址配置  其他地址 可行的
https://kfwkfulq.mirror.aliyuncs.com
https://2lqq34jg.mirror.aliyuncs.com
https://pee6w651.mirror.aliyuncs.com

```shell
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": [
    "https://kfwkfulq.mirror.aliyuncs.com",
    "https://2lqq34jg.mirror.aliyuncs.com"
  ]
}
EOF
sudo systemctl daemon-reload && sudo systemctl restart docker
```

# 运行 docker 容器
```shell
# 创建数据目录（宿主机目录，用于持久化存储）
mkdir -p ~/minio/data

# 运行容器（指定端口映射、数据卷和密钥）
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  -v ~/minio/data:/data \
  -e "MINIO_ROOT_USER=minio" \
  -e "MINIO_ROOT_PASSWORD=minio028" \
  minio/minio server /data --console-address ":9001"
```

# 访问地址
http://8.137.18.24:9001/