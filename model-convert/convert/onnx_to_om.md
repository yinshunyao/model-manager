在使用华为昇腾（Ascend）AI处理器的 **ATC（Ascend Tensor Compiler）** 工具进行模型转换时，`--input_shape` 参数用于**指定模型输入张量的静态形状（shape）**，这是 ATC 编译过程中**最关键且最容易出错的参数之一**。

---

## ✅ 一、`--input_shape` 到底填什么？

### 简单回答：
> **填你的模型在推理时实际输入的 tensor shape，去掉 batch 维度后的具体数值（或保留 batch 用 -1 / 动态 shape）**，格式为：  
> ```
> --input_shape="input_name:shape"
> ```

但细节取决于你的模型类型和是否使用动态 shape。

---

## 🔍 二、详细说明

### 1. **基本格式**
```bash
atc --model=xxx.om \
    --framework=5 \
    --input_shape="input_tensor_name:1,3,224,224" \
    --output=xxx
```

- `input_tensor_name`：模型输入节点的名称（必须与模型中一致）
- `1,3,224,224`：表示 `[batch, channel, height, width]`（NCHW 格式）

> ⚠️ 昇腾默认使用 **NCHW** 数据布局（与 PyTorch 一致，不同于 TensorFlow 的 NHWC）

---

### 2. **如何获取 input_tensor_name？**

#### 方法一：用 Netron 查看 `.onnx` 模型
- 打开 ONNX 模型 → 查看 inputs → 名称如 `"input"`、`"images"`、`"x"` 等

#### 方法二：用 Python 脚本打印
```python
import onnx
model = onnx.load("model.onnx")
for inp in model.graph.input:
    print("Input name:", inp.name)
    # 打印 shape（可能是动态的，如 ['batch', 3, 224, 224]）
```

---

### 3. **常见场景示例**

| 场景 | `--input_shape` 示例 | 说明 |
|------|---------------------|------|
| **图像分类（固定 batch=1）** | `"input:1,3,224,224"` | 最常见 |
| **目标检测（YOLO）** | `"images:1,3,640,640"` | 注意输入名可能不是 "input" |
| **NLP（BERT）** | `"input_ids:1,128", "attention_mask:1,128"` | 多输入需全部指定 |
| **动态 batch（推荐）** | `"input:-1,3,224,224"` | `-1` 表示 batch 可变（需配合 `--dynamic_batch_size`） |
| **完全动态 shape（谨慎）** | `"input:-1,-1,-1,-1"` | 性能差，仅调试用 |

---

### 4. **动态 Shape 支持（重要！）**

昇腾 910B 支持动态 shape，但需正确配置：

#### ✅ 推荐方式：动态 batch
```bash
--input_shape="input:-1,3,224,224" \
--dynamic_batch_size="1,2,4,8"
```
- 实际推理时 batch 必须是 `1/2/4/8` 中的一个
- ATC 会预编译多个 batch 的 kernel，提升灵活性

#### ❌ 不推荐：全动态
```bash
--input_shape="input:-1,-1,-1,-1"
```
- 性能严重下降（无法做算子融合优化）
- 仅用于调试或 shape 完全未知的场景

---

### 5. **PyTorch 导出 ONNX 时的注意事项**

确保导出时**明确指定输入 shape**，避免动态符号：

```python
# ❌ 错误：使用 None 会导致 ONNX 中为 dynamic
torch.onnx.export(model, torch.randn(1,3,224,224), ...)

# ✅ 正确：用具体数值
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx", ...)
```

如果 ONNX 中已有动态维度（如 `batch`），ATC 仍可通过 `--input_shape` 覆盖为静态或半动态。

---

## 🛠️ 三、常见错误 & 解决方案

| 错误信息 | 原因 | 解决 |
|--------|------|------|
| `Input shape mismatch` | 输入名或 shape 与模型不符 | 用 Netron 确认输入节点名和维度 |
| `The shape contains invalid value` | 用了 0 或负数（除 -1 外） | 检查 shape 数值 |
| `Dynamic shape not supported for operator XXX` | 某些算子不支持动态 shape | 改为静态 shape 或升级 CANN 版本 |
| `Batch size must be fixed` | 模型含不支持动态 batch 的算子 | 使用 `--dynamic_batch_size` 而非全动态 |

---

## ✅ 四、最佳实践建议（针对 910B）

1. **优先使用静态 shape**（如 `1,3,224,224`）以获得最佳性能
2. **若需多 batch 支持**，用 `--dynamic_batch_size="1,2,4,8"` + `input:-1,...`
3. **输入名必须精确匹配** ONNX 模型中的 input name
4. **数据布局统一为 NCHW**（PyTorch 默认，TensorFlow 需转置）
5. **CANN 版本 ≥ 7.0** 对动态 shape 支持更好

---

## 🔗 附：ATC 官方文档参考
- [华为 CANN ATC 工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1alpha001/infacldevg/atctool/atlasatc_16_0001.html)

---

如果你提供：
- 模型类型（ResNet/YOLO/BERT？）
- ONNX 输入节点名
- 期望的 batch size 范围

我可以帮你写出**完整的 ATC 命令**。