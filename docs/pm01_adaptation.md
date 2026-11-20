# PM01 适配：把机器人、参考和执行时序对齐

PM01 版本沿用 SuperMimic 的 Actor、F、Critic、replay 和想象优化流程，主要工作是把机器人规格、数据来源、观测、参考及命令时序接入同一套训练与推理契约。本文说明实现中的适配点；这些工程改动不作为原创 SuperMimic 算法声明。

*English: This adaptation preserves the SuperMimic learning loop while aligning PM01 joint names, exogenous head control, reference encoding, source-aware sampling, and command-delay state across collection, replay, model fitting, and imagination.*

全局关系见[架构说明](architecture.md)，成绩口径见[评测说明](evaluation.md)。公开目录是维护版本，包含后续修复；65k 快照与当前源码的关系见 [`training/source_provenance.json`](../training/source_provenance.json)。

## 1. 24 个模型关节，23 个学习动作

| 契约 | PM01 设置 |
| --- | --- |
| 基座 | `LINK_BASE` |
| 模型关节 | 具名的 24 个关节，顺序固定 |
| Actor 输出 | 前 23 个身体关节 |
| 外生关节 | 最后一维 `J23_HEAD_YAW`，取当前参考角 |
| 被跟踪身体 | 固定的 13 个身体，用于运动学、奖励及原生终止 |
| 策略/参考频率 | 50 Hz，20 ms 一拍 |
| 训练物理频率 | 200 Hz，5 ms 子步，每拍 4 个子步 |

“外生头部”表示头部目标不由 Actor 的 23 维输出学习产生；头部仍存在于仿真状态、本体观测、参考、完整 24 维命令和延迟状态中。它不应被删除，也不能给 Actor 额外补一个学习输出而沿用同一权重。

动作转换为：

```text
head_action = (current_reference_head − default_head − calibration_head) / scale_head
action24 = clip(concat(action23, head_action), −20, 20)
q_target = default_q + calibration + action_scale × action24
```

仿真模型使用名义 PD 跟踪该目标，并按模型限制产生力矩。离线核心只返回目标，不执行 PD。默认角、缩放、增益和力矩上限来自模型规格，不能将 `[-20,20]` 的动作裁剪理解为实机安全保护或关节限位检查。

关节、身体和外生头部约束由 [`spec.py`](../training/holomotion/src/env/reference_tracking/spec.py) 显式检查；公开推理数值在 [`robot_spec.json`](../inference/robot_spec.json)。完整 MJCF/URDF/mesh 不随首版分发。

## 2. 本体观测与未来参考

### 78 维本体观测，10 帧历史

| 顺序 | 字段 | 维数 |
| --- | --- | ---: |
| 1 | `q − default_q − calibration` | 24 |
| 2 | `q̇` | 24 |
| 3 | LINK_BASE 坐标系的角速度 | 3 |
| 4 | LINK_BASE 坐标系的单位重力方向 | 3 |
| 5 | 上一拍实际发出的组合动作 | 24 |
| 合计 | 每帧本体观测 | 78 |

历史按最旧到最新排列；初始帧复制填满 10 帧。关节角用 rad，关节及基座角速度用 rad/s。部署输入须先转到模型关节顺序与 LINK_BASE 坐标约定。输入四元数为 world-from-LINK_BASE、`wxyz`，重力投影为 `Rᵀ × [0,0,−1]`。

Actor 不直接接收实际根位置、根线速度或当前根跟踪误差。训练的 F、Critic 和计分函数使用的仿真状态更丰富，不能照搬为部署输入。观测构造见 [`MDP.obs`](../training/holomotion/src/env/reference_tracking/mdp.py) 与 [`PolicyCore.observe`](../inference/supermimic_pm01/policy_core.py)。

### 630 维未来目标，使用原参考

取当前参考帧之后 `[5,10,15,20,25,30,35,40,45,50]` 帧，即 0.1 至 1.0 秒。每个未来帧 63 维，10 帧合计 630 维：

| 字段 | 表示坐标与含义 | 维数 |
| --- | --- | ---: |
| 根位置差 | 未来参考减当前参考，转到当前参考航向坐标系 | 3 |
| 根旋转 | 未来参考旋转相对当前参考航向；旋转矩阵前两列按行展开 | 6 |
| 根线速度 | 未来参考根部自身坐标系 | 3 |
| 根角速度 | 未来参考根部自身坐标系 | 3 |
| 关节角 | 未来 24 维参考角 | 24 |
| 关节速度 | 未来 24 维参考速度乘 `0.1` | 24 |

越过片尾时重复最后参考帧。头部控制取的是**当前**参考头角，不能用最远未来头角代替。目标里的“位置差”是两个参考时刻之间的差，不是“未来参考减实际机器人位置”。本轮默认 `reference_only_v0`，不使用历史实验中的 `goal_feedback_v1`。

参考读取使用具名 24 关节、50 Hz HDF5 契约，检查关节顺序、帧率和输入 `xyzw` 四元数声明，再转成内部表示。公开离线 `ReferenceClip` 使用已编码的 `q` 与 `goal`；首版合成示例不包含真实参考动作。编码定义见 [`MotionLib.from_hdf5 / goal`](../training/holomotion/src/env/reference_tracking/motion.py)，推理接口见 [`inference/README.md`](../inference/README.md)。

## 3. 0–20 ms 延迟贯穿学习链路

只在物理引擎中推迟目标、却让 F 假设新目标立即生效，会让预测条件与真实转移不一致。PM01 在 20 ms 控制拍内显式处理四个 5 ms 子步；训练每回合从 `0/5/10/15/20 ms` 抽取一个延迟并保持。

延迟状态为 49 维：

```text
d = concat(当前保持的 24 维目标, 最近发出的 24 维目标, 延迟子步数)
延迟子步数 ∈ {0,1,2,3,4}
```

令新目标为 `u[t]`，上一拍发出目标为 `u[t−1]`，延迟子步数为 `k`。当前拍第 `j=0,1,2,3` 个子步在 `j<k` 时使用旧目标，否则使用新目标。例如 10 ms 延迟：

| 当前控制拍的物理区间 | 0–5 ms | 5–10 ms | 10–15 ms | 15–20 ms |
| --- | --- | --- | --- | --- |
| 实际执行目标 | `u[t−1]` | `u[t−1]` | `u[t]` | `u[t]` |

0 ms 在首个子步使用新目标；20 ms 在本拍四个子步中都使用上一目标，新目标到下一拍才生效。等待期间物理仿真照常推进。

```mermaid
flowchart LR
    A[23 维 Actor 动作 + 当前参考头部] --> Q[24 维关节目标]
    D[命令状态 d] --> P[按 5 ms 子步选择目标]
    Q --> P
    P --> S[MuJoCo-Warp 实际下一状态]
    S --> B[Replay：s、a、s1 与本拍之前的 d]
    D --> B
    B --> F[F 拟合：状态特征 + 动作 + 物理上下文 + d]
    B --> I[想象起点：恢复 d]
    F --> I
    I --> U[同一解析规则推进 d，交给下一想象步]
```

接口的一致性体现在：

- **采集：** 真正按子步切换目标，保存推进前的队列与实际下一状态；回合重置前先保存转移。
- **Replay / F：** replay 保留 49 维状态，F 的每步预测显式读取相应的队列；F 拟合用记录的物理下一状态作监督。
- **想象：** 从 replay 恢复队列，F 预测下一状态后，用同一解析函数推进命令状态。队列推进对动作可微，Actor 更新可获得延迟下的后果梯度。
- **Critic：** 估值时读取同时刻的队列状态。
- **Reset：** 用重置后的实际关节姿态初始化保持目标与最近发出目标。训练时身体关节角已加入初态扰动，干净评测时才是参考姿态；重新抽取该回合的延迟，不沿用上个回合队列。

这 49 维没有增加到 Actor 的 78 维观测中。Actor 历史中的上一动作是“已发出动作”，不等于延迟后当前生效的目标，也不能用实测关节角替代。当前公开 `PolicyCore` 不模拟延迟；执行层和仿真器需各自实现对应时序。

实现见 [`command_delay.py`](../training/holomotion/src/env/reference_tracking/command_delay.py)、[`warp_backend.py`](../training/holomotion/src/env/reference_tracking/warp_backend.py)、[`buffer.py`](../training/holomotion/src/algo/supermimic/buffer.py) 和 [`learner.py`](../training/holomotion/src/algo/supermimic/learner.py)。0/5/10/15/20 ms 的回合固定设置参考了 [MorphMimic 固定版本](https://github.com/lzyang2000/morphmimic/blob/770394b4a0cad9effbf807a6c03435d1699733d0/src/morphmimic/env/env_cfg.py#L179)；这里说明工程来源，不据此宣称 PM01 具备同等真机表现。

## 4. 来源预算与困难采样分两层

本轮训练池共 3,982 条片段/变体，采样器按下表设置**重置起点的目标抽取份额**：

| 来源组 | 片段/变体数 | 目标份额 |
| --- | ---: | ---: |
| 通用动作 | 3,950 | 90% |
| 跑步/过渡 | 4 条原参考 + 4 条变体 | 5% |
| 跳跃 | 12 条原参考 + 12 条变体 | 5% |

组内把参考切成约 1 秒的有效起点区间，使用沿用配方的 `50% 均匀 + 40% 当前困难 + 10% 历史困难` 混合。它与组间的 90/5/5 表示不同的量。

```mermaid
flowchart LR
    G[来源组预算：90 / 5 / 5] --> B[组内来源时间区间：50 / 40 / 10 混合]
    B --> V[选择对应原参考或镜像变体]
    V --> F[在区间中选有效参考帧]
    F --> R[恢复参考状态并加入训练扰动]
```

图表示等价的采样逻辑，源码将各部分合成最终权重后一次抽样。当前困难分布为 `softmax(−25×G/G_max)`；历史分布按 `0.999×旧分布 + 0.001×当前分布` 更新。回合统计先累加实际仿真采集的折扣奖励；若因自然片尾或时间上限结束、未触发物理终止，再加 `γ^T × 回合平均一步奖励 / (1−γ)` 的长度补估，物理终止时该尾项为零。这项补估不调用 Critic。

回合结束时更新其起始区间：同区间并发回报先求均值，随后按 `G = 0.8×旧值 + 0.2×组内平均回报` 更新。这里的 `G` 不是 Actor/Critic loss，也不是完整评测的 `R_norm`；不能把它和无折扣、按计划全长归一化的评测回报混称。

同一源时间区间的原参考与镜像/小幅修正版共享困难统计，扩展到变体时按变体数分摊概率。这样，增加一份镜像不会只因文件数量翻倍而自动翻倍该来源区间的份额。并行回合命中同一区间时先聚合，再更新回报统计。

90/5/5 不是保证的真实执行帧比例或梯度比例：不同回合长度、失败时刻和 replay 重用都会改变后续数据分布。F 窗口和 Actor 起点从 replay 抽取，运行时另外记录来源计数。也不能将镜像、同源切窗或两个重复评测当作独立录制或不同训练 seed。

权重实现见 [`SourceBinSampler`](../training/holomotion/src/env/reference_tracking/source_sampling.py)，采集及 F/Actor 使用计数见 [`retrain_runtime.py`](../training/holomotion/src/algo/supermimic/retrain_runtime.py)。这条 PM01 训练路径使用前者，不能仅因仓库中存在通用 `source_adaptive_sampling.py` 就认定本轮使用了另一套采样器。

## 5. 物理扰动、片尾和部署接入各有边界

训练使用的物理上下文为 `ξ`：摩擦系数、24 维关节校准偏移和躯干质心偏移，共 28 维。F 与 Critic 显式读取归一化后的上下文。训练还对本体观测加噪声，并从有效参考帧恢复根/关节状态后加入初态扰动；头部保留外生参考角。原生干净评测关闭观测噪声、使用固定物理上下文，执行延迟则按各评测协议指定；具体设置见[评测说明](evaluation.md)。

训练可从正在跑或跳的参考中途恢复相应位置、姿态与速度。这不要求每条训练片段都从站定开始，也不自动解决机器人从站定进入跑跳的接管问题。参考自然片尾、物理失败与完整技能的安全退出同样需要区分：F 窗口不跨回合，Actor 的自然片尾 mask 保留最后有效奖励，不能把片尾后重复的参考当成新学习转移。

公开维护版还保留了 MuJoCo-Warp EPA 容量及 CUDA 捕获缓冲生命周期相关修补，并在发布整理时替换了受限的通用数学辅助实现。这些是运行与发布维护工作，不是新的 Actor 学习算法。修补范围和复现缺口见 [`training/README.md`](../training/README.md)。

首版可运行的是离线推理检查；完整数据、物理资产、完整学习恢复点和现场执行桥未随包提供。仿真完成、FP32 导出数值通过和受控 1 秒探针分别证明不同事情，持续真机静站与完整敬礼仍未验收。数据归属和许可见[模型卡](model_card.md)、[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)与[发布范围](release_scope.md)。
