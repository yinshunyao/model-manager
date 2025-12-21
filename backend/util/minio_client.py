from minio import Minio
from minio.error import S3Error

# MinIO 配置
MINIO_ENDPOINT = "10.13.41.151:9000"  # 不要加 http://
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "miniominio"
MINIO_SECURE = False  # 开发环境使用 HTTP
BUCKET_ONNX = "onnx-file"
BUCKET_MODEL = "model-file"
BUCKET_SOURCE = "source-file"
BUCKET_TARGET = "target-file"
BUCKET_ENGINE = "engine-file"

# 初始化 MinIO 客户端
client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

# 确保 bucket 存在
def ensure_bucket_exists():
    try:
        if not client.bucket_exists(BUCKET_ONNX):
            client.make_bucket(BUCKET_ONNX)
            print(f"Bucket '{BUCKET_ONNX}' created.")
        else:
            print(f"Bucket '{BUCKET_ONNX}' already exists.")

        if not client.bucket_exists(BUCKET_MODEL):
            client.make_bucket(BUCKET_MODEL)
            print(f"Bucket '{BUCKET_MODEL}' created.")
        else:
            print(f"Bucket '{BUCKET_MODEL}' already exists.")

        if not client.bucket_exists(BUCKET_SOURCE):
            client.make_bucket(BUCKET_SOURCE)
            print(f"Bucket '{BUCKET_SOURCE}' created.")
        else:
            print(f"Bucket '{BUCKET_SOURCE}' already exists.")

        if not client.bucket_exists(BUCKET_TARGET):
            client.make_bucket(BUCKET_TARGET)
            print(f"Bucket '{BUCKET_TARGET}' created.")
        else:
            print(f"Bucket '{BUCKET_TARGET}' already exists.")

        if not client.bucket_exists(BUCKET_ENGINE):
            client.make_bucket(BUCKET_ENGINE)
            print(f"Bucket '{BUCKET_ENGINE}' created.")
        else:
            print(f"Bucket '{BUCKET_ENGINE}' already exists.")
    except S3Error as e:
        print(f"Error creating bucket: {e}")

# 调用一次即可
# ensure_bucket_exists()