# erlang
## 确认ubuntu版本号
```shell
lsb_release -a
```

## 安装erlang
⚠️ 使用如下脚本安装，注意：将 jammy 替换为你系统的代号：
Ubuntu 20.04 → focal
Ubuntu 22.04 → jammy
Ubuntu 24.04 → noble

```shell
# 仓库配置，不是必须
# 下载并添加 Erlang Solutions 的 GPG 密钥
# wget -O- https://packages.erlang-solutions.com/ubuntu/erlang_solutions.asc | sudo gpg --dearmor -o /usr/share/keyrings/erlang.gpg

# 添加仓库（以 Ubuntu 22.04 为例，代号 jammy；如果是 20.04 则是 focal，24.04 是 noble）
# echo "deb [signed-by=/usr/share/keyrings/erlang.gpg] https://packages.erlang-solutions.com/ubuntu jammy contrib" | sudo tee /etc/apt/sources.list.d/erlang.list
# 更新包列表
# sudo apt update

# 安装 Erlang
sudo apt install -y erlang-base erlang-nox erlang-dev erlang-src

```


## erlang 验证安装
```shell
erl -version
```


# rabbitmq安装
```shell
# 添加 RabbitMQ 签名密钥
# sudo apt install -y curl gnupg
# curl -fsSL https://github.com/rabbitmq/signing-keys/releases/download/3.0/rabbitmq-release-signing-key.asc | sudo gpg --dearmor -o /usr/share/keyrings/rabbitmq.gpg

# 添加 RabbitMQ APT 仓库
# echo "deb [signed-by=/usr/share/keyrings/rabbitmq.gpg] https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-server/deb/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/rabbitmq.list

# 更新并安装
# sudo apt update
sudo apt install -y rabbitmq-server
```

# 启动rabbitmq
```shell
# 启动服务
sudo systemctl start rabbitmq-server

# 设置开机自启
sudo systemctl enable rabbitmq-server

# 检查状态
sudo systemctl status rabbitmq-server
```

# 启用管理插件（Web UI）
```shell
sudo rabbitmq-plugins enable rabbitmq_management
```

# 创建管理员用户
```shell
# 添加新用户（例如 admin）
sudo rabbitmqctl add_user admin rmq-sd-sc

# 设置为管理员角色
sudo rabbitmqctl set_user_tags admin administrator

# 授予所有权限（vhost "/"）
sudo rabbitmqctl set_permissions -p / admin ".*" ".*" ".*"
```


# 防火墙设置（如启用 UFW）
```shell
sudo ufw allow 5672/tcp   # AMQP 协议端口
sudo ufw allow 15672/tcp  # Web 管理界面
sudo ufw reload
```

# 登录web删除guest账号
http://xxx.xx.xx.xx:15672/#/users

# 常用命令


|功能|命令|
|---|---|
|查看用户|	sudo rabbitmqctl list_users|
|删除用户|	sudo rabbitmqctl delete_user username|
|查看队列|	sudo rabbitmqctl list_queues|
|查看连接	|sudo rabbitmqctl list_connections|
|重启服务	|sudo systemctl restart rabbitmq-server|

# 配置文件（可选）
默认配置文件路径：
- 主配置：/etc/rabbitmq/rabbitmq.conf
- 环境变量：/etc/rabbitmq/rabbitmq-env.conf