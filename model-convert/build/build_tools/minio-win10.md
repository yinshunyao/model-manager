# Windows 10 安装 MinIO 简易文档
MinIO 是一款高性能的对象存储服务，兼容 Amazon S3 API，适合本地开发、测试或小规模部署。以下是 Windows 10 系统下 MinIO 的快速安装和基础使用步骤。

## 一、环境准备
1. **系统要求**：Windows 10 64位（32位系统需下载对应版本，但官方优先推荐64位）。
2. **无需预装依赖**：MinIO 为单二进制文件，直接运行即可，无需安装Java/Go等环境。

## 二、下载 MinIO 安装包
1. 打开 MinIO 官方下载地址：  
   [MinIO Windows 下载页](https://min.io/download#/windows)  
   或直接下载64位二进制包（推荐）：  
   `https://dl.min.io/server/minio/release/windows-amd64/minio.exe`
2. 下载完成后，将 `minio.exe` 放到自定义目录（建议路径简洁，如 `D:\minio`），避免中文/空格路径。

## 三、启动 MinIO 服务
### 方式1：临时启动（控制台关闭即停止）
1. 打开 **命令提示符（CMD）** 或 **PowerShell**（以管理员身份运行更佳）。
2. 切换到 `minio.exe` 所在目录，例如：  
   ```bash
   cd D:\minio
   ```
3. 启动 MinIO 服务，指定数据存储目录（需提前创建，如 `D:\minio\data`）：  
   ```bash
   minio.exe server D:\minio\data --console-address ":9001"
   ```
   - 说明：
     - `D:\minio\data`：MinIO 存储数据的目录，可自定义；
     - `--console-address ":9001"`：指定 Web 控制台端口（默认9000易冲突，建议改9001）；
     - 启动成功后，控制台会输出访问地址、Access Key、Secret Key（默认均为 `minioadmin`）。

### 方式2：后台启动（常驻服务，推荐）
若需 MinIO 开机自启/后台运行，可借助 `nssm` 工具将其注册为 Windows 服务：
1. 下载 `nssm`：[NSSM 下载页](https://nssm.cc/download)，解压后将 `nssm.exe` 放到 `D:\minio` 目录。
2. 以管理员身份打开 CMD，执行以下命令注册服务：
   ```bash
   # 切换到minio目录
   cd D:\minio
   # 注册服务（服务名MinIO，可自定义）
   nssm install MinIO "D:\minio\minio.exe" "server D:\minio\data --console-address :9001"
   ```
3. 启动服务：
   ```bash
   nssm start MinIO
   ```
4. 管理服务（可选）：
   - 停止：`nssm stop MinIO`
   - 删除：`nssm remove MinIO confirm`

## 四、访问 MinIO 控制台
1. 打开浏览器，访问地址：`http://localhost:9001`（或 `http://本机IP:9001`，如 `http://192.168.1.100:9001`）。
2. 输入默认账号密码：
   - Access Key：`minioadmin`
   - Secret Key：`minioadmin`
3. 登录后即可创建 Bucket、上传文件、配置权限等操作。

## 五、基础配置（可选）
### 1. 修改默认账号密码
启动时通过环境变量指定自定义密钥（避免默认密码泄露）：
```bash
# CMD 中设置临时环境变量
set MINIO_ROOT_USER=your_username
set MINIO_ROOT_PASSWORD=your_password
# 再启动服务
minio.exe server D:\minio\data --console-address ":9001"
```
（若为 Windows 服务，需在 nssm 中配置环境变量：`nssm set MinIO Environment MINIO_ROOT_USER=your_username;MINIO_ROOT_PASSWORD=your_password`）

### 2. 配置端口
- 数据端口（S3 API）：默认9000，可通过 `--address ":自定义端口"` 修改，例如：
  ```bash
  minio.exe server D:\minio\data --address ":8080" --console-address ":9001"
  ```

## 六、常见问题
1. **端口被占用**：修改 `--console-address` 或 `--address` 后的端口号（如改为9002、8081）。
2. **权限不足**：以管理员身份运行 CMD/PowerShell，或确保数据目录（如 `D:\minio\data`）有读写权限。
3. **无法访问控制台**：检查防火墙是否放行对应端口，或使用本机IP（而非localhost）访问。

## 七、停止 MinIO
1. 临时启动：在 CMD/PowerShell 中按 `Ctrl+C` 停止。
2. 服务启动：执行 `nssm stop MinIO`。