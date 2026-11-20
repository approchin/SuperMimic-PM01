# PM01 训练与评测源码

此目录是从已有 HoloMotion/SuperMimic 工作树选出的 PM01 维护版本，提供 Actor、世界模型 F、Critic、想象优化、MuJoCo-Warp 环境、执行延迟、采样与评测实现。首版的可直接运行入口是仓库根目录说明的离线推理。

这里不包含完整训练数据、评测参考、机器人 MJCF/URDF/mesh 或完整恢复点。仅有本目录不能重现整轮训练，也不能用轻量 Actor 恢复优化器、经验池及世界模型。

## 阅读顺序

1. `train_011.py`：本轮启动包装；通用入口是 `holomotion/scripts/training/train_supermimic.py`。
2. `holomotion/src/algo/supermimic/networks.py`、`learner.py`、`imagination.py`：网络与更新。
3. `holomotion/src/env/reference_tracking/spec.py`、`mdp.py`、`command_delay.py`、`warp_backend.py`：PM01 契约和物理环境。
4. `holomotion/src/algo/supermimic/retrain_runtime.py`、`retrain_eval_worker.py`：运行策略和原生评测。

从仓库根目录可只查看参数帮助，不启动训练：

```bash
python training/train_011.py --help
```

使用源码模块时，将 `training/` 加入 Python 模块搜索路径；导出工具的 `--training-code training` 会完成这一设置。此快照不声称提供完整 HoloMotion 的其他任务、Isaac 后端或 G1 工作流。

## 65k 权重对应的已记录配方

以下来自该 Actor 的训练规格；它们不是通用入口默认值，也不是当前机器已重新运行的记录。

| 参数 | 本轮记录 |
| --- | --- |
| 随机种子 | 13，单一主训练模型 |
| 并行环境 | 5,120 |
| Actor 逻辑 batch / microbatch | 4,096 / 4,096 |
| UTD 调度事件率 | 每 vector step 为 1/2；不能当作每条 transition 的比率 |
| buffer steps | 1,000 |
| 优化学习率 | 2e-4 |
| 数值与后端 | CUDA BF16 MLP，MuJoCo-Warp；物理与归一化保留 FP32 |
| 执行延迟 | 每回合固定抽取 0/5/10/15/20 ms，进入 F、经验与想象的一致接口 |
| 图捕获与编译 | 原运行均启用 |
| 原运行依赖 | Python 3.10.19、Torch 2.9.1+cu128、MuJoCo 3.16.0、MuJoCo-Warp 3.16.0、Warp 1.18.0 |

`requirements.txt` 是历史训练依赖记录，不是离线推理所需环境。首版的 CPU 推理及重新导出使用的本地环境与历史训练环境不同，不能据此宣称训练环境已复现。

## 源码版本与公开副本改动

HoloMotion 基线为 `f52dd81ae935a828b0296f5ee0f833beb7378bf0`，紧凑 SuperMimic 参考实现的固定版本见模块内 `upstream.json`。本目录保留后期恢复和评测修复，不能称为生成 65k 权重时的逐字源码。已归档 Actor 的 49 项源码指纹中，43 项与本机维护工作树相符；另外六项涉及检查点恢复、评测队列、执行延迟和 CUDA 缓冲生命周期维护。可公开的对应关系见 [`source_provenance.json`](source_provenance.json)。

公开整理额外作了这些更改：

- 添加 PM01 修改标记，保留上游版权、两份 MIT 文本和 `NOTICE`。
- 不分发历史环境中带特殊再分发限制的 `torch_utils.py`；为数据加载/FK 使用的 11 个数学接口编写 `rotation_math.py`，两处导入相应调整。
- 调整可选 Isaac 评测组件的导入时机，使阅读 PM01 模块不依赖未分发的可选后端。
- 将公开数学替代和来源采样模块纳入之后运行时的源码指纹记录。

数学替代已在 CPU 上对 SciPy 检查 1,030 个随机与边界旋转、组合、向量旋转和时间差分。运行入口为 `python tools/check_rotation_math.py`，需要 Torch、SciPy、NumPy。这个检查不验证完整训练、GPU 编译或原生物理性能。

## 已安装依赖的历史修补

`patches/restore_native_epa64.py` 记录原运行对 MuJoCo-Warp 3.16.0 EPA 容量的修补。它只面向明确指定的独立 `.venv`，先核对版本与文件 SHA256，再更改唯一常量；不是通用修补脚本，不应在共享环境中直接执行。

66k 后的 CUDA 捕获缓冲生命周期修复已存在于这个维护快照，评测精度和版本边界另见 [`docs/evaluation.md`](../docs/evaluation.md)。首版没有启动新的训练或 GPU 评测；完整复现还需逐项对齐数据、资产、依赖、补丁和最终参数。
