

# 

# 转换

atc   --model=/root/miniconda3/convert/predict/../model_demo/yolo11n.onnx  --framework=5  --output=/root/miniconda3/convert/predict/../model_demo/model_test   --soc_version=Ascend910B2 --log=info --input_format=NCHW --output_type=FP32 --input_shape="images:1,3,640,640"   --output="output0" --op_select_implmode=high_precision  
