# 65k 离线推理

本目录提供独立的 CPU ONNX Runtime 推理核心：接收模型坐标系下的状态和已编码参考，输出动作与关节目标。核心不连接机器人、仿真器或网络。仓库不提供可直接用于实机的运行入口；此前持续静站和完整敬礼尚未通过真机验收。

## 文件和检查

按[项目首页的权重入口](../README.md#4-权重与离线使用)取得两个模型文件，将它们放入 `models/`，在仓库根目录运行。

```bash
python -m pip install -r inference/requirements.txt
python tools/verify_weights.py
python tools/smoke_inference.py
```

`verify_weights.py` 用标准库核对两个文件的大小和 SHA-256；提供 `--tensors` 时还用 PyTorch 逐一核对 26 个 Actor 和归一化张量的名称、形状、dtype 与内容。校验和只能检查文件与清单是否一致，不能独立证明发布来源。

`smoke_inference.py` 构造静止状态和合成参考，在 CPU 上运行两步、重置并对比独立 PyTorch FP32 输出。它不需要动作数据、GPU 或机器人，也不验证闭环动作表现。已在 Python 3.11.15、NumPy 1.26.4、ONNX Runtime 1.23.2 的 CPUExecutionProvider 上运行；其他版本未逐一验证。

`actor_65000_fp32.onnx` 是原 FP32 导出，观测和参考归一化包含在图中；原训练评测的 MLP 使用 CUDA BF16，因此不能把 ONNX FP32 结果直接当作该 BF16 的闭环结果。`actor_0000065000.pt` 是原 65k 轻量 Actor checkpoint 的**元数据脱敏副本**：保留了全部策略和归一化张量、导出所需结构及公开归属信息，私有路径、动作名称和机器标识等元数据已清理；它不支持恢复训练。发布文件 SHA-256 因脱敏而不同。ONNX 内和 `actor_65000.json` 中的 `source_actor_sha256` 指向脱敏前的原始 checkpoint，该原始文件不在本包中。公开副本的摘要另见 `published_actor_sha256` 和 `SHA256SUMS.json`；各原始张量的内容指纹在 `actor_tensor_sha256.json` 中。

`robot_spec.json` 是从原模型规格提取的必要数值字段。它保留训练使用的默认关节角、动作缩放及名义增益，不应视为已经验收的真机参数。动作素材和机器人 mesh 均未随本包提供。

## 接口契约

ONNX 单次推理要求 FP32、固定 batch=1：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `obs_hist` | `[1, 10, 78]` | 本体观测，从最旧到最新；首次观测重复 10 次 |
| `goal` | `[1, 630]` | 10 个未来参考帧，各 63 维 |
| `action23` | `[1, 23]` | Actor 的 23 维身体动作，尚未裁剪 |

每帧本体观测依次为关节角相对默认角及校准值 `q - default_q - calibration`（24）、关节角速度（24）、LINK_BASE 坐标系的角速度（3）、LINK_BASE 坐标系中投影的单位重力方向（3）、上一拍已发出的组合动作（24）。关节顺序见 `robot_spec.json`；关节角单位为 rad，速度为 rad/s。姿态输入是 world-from-LINK_BASE 四元数，顺序为 `wxyz`。

参考在 50 Hz 下取当前帧之后的 `+5, +10, …, +50` 帧，超出片尾时重复最后一帧。每个未来帧依次包含当前参考航向坐标系中的根位置差（3）、未来参考姿态相对当前参考航向的旋转矩阵前两列按行展开（6）、未来参考根部自身坐标系中的线速度与角速度（各 3）、未来 24 关节角（24）、未来 24 关节角速度乘 `0.1`（24）。本包不提供真实参考动作或其转换数据；输入须使用同一坐标、顺序和编码约定。训练实现中的 `MotionLib.goal` 定义可供对照。

`PolicyCore` 将 23 维输出与**当前参考帧**的头部 yaw 组合，裁剪到 `[-20, 20]`，再按 `default_q + calibration + action_scale * action24` 生成 24 维目标。上次发出的组合动作进入下一帧观测；若外部处理修改目标，应在下一次 `step()` 前用 `commit_issued_target()` 写回实际目标。此接口不执行 PD 控制或限速。

`ReferenceClip` 的数组应为 `q: [N, 24]`、`goal: [N, 630]`，帧率 50 Hz，关节顺序与 `JOINT_NAMES` 一致。也可用 `ReferenceClip.load()` 读取包含 `q`、`goal`、`fps`、Unicode `joint_names` 的 NPZ；需禁用 pickle，帧数至少为 2。合成构造及完整 `PolicyCore` 调用示例见 `tools/smoke_inference.py`。

## 从 checkpoint 重导出（可选）

CPU 推理只需 ONNX Runtime；重导出另需与公开训练源码兼容的 PyTorch、ONNX 和训练代码导入依赖。请使用新输出目录，避免覆盖已验证模型：

```bash
python inference/export_onnx.py \
  --checkpoint models/actor_0000065000.pt \
  --training-code training \
  --spec inference/robot_spec.json \
  --models local_export
```

已用公开训练树和脱敏 checkpoint 在 CPU 上重导出，并对五组合成输入做 FP32 数值对照。PyTorch/ONNX 版本不同可能使导出文件的字节摘要不同；新导出的 metadata 中源 checkpoint 摘要指向传给导出器的脱敏副本，而原始已发布 ONNX 的源摘要指向脱敏前 checkpoint。
