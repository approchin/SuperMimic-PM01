# 第三方来源、署名与许可范围

项目代码采用 [Apache-2.0](LICENSE)。原文件的第三方许可和版权声明继续适用。模型权重采用[非商业科研模型许可 v0.1](models/LICENSE.md)，视频、截图、结果图及统计采用[媒体科研使用条款](media/LICENSE.md)。

## HoloMotion 与紧凑 SuperMimic 实现

训练代码来自 [HorizonRobotics/HoloMotion](https://github.com/HorizonRobotics/HoloMotion)，记录的集成基线为 `f52dd81ae935a828b0296f5ee0f833beb7378bf0`。Apache-2.0 原文保存在 [`LICENSES/HoloMotion-Apache-2.0.txt`](LICENSES/HoloMotion-Apache-2.0.txt)，上游综合声明保存在根目录和训练目录的 `NOTICE`。

部分紧凑网络、缓冲、运动学与想象更新实现来自 [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed/tree/a1508d42c9d1699a52e8233ab16d08c488441ab9)，固定版本为 `a1508d42c9d1699a52e8233ab16d08c488441ab9`，保留 MIT 全文、模块内两份 `LICENSE.vibed` 和 [`upstream.json`](training/holomotion/src/algo/supermimic/upstream.json)。算法论文为 [SuperMimic](https://arxiv.org/abs/2610.28480)。本项目负责 PM01 适配与相关工程修改。

上游 `NOTICE` 同时提及 ASAP 和 omomo_release 的 MIT 来源，原文继续保留。它也列出了历史 NVIDIA 工具的单独限制；该历史 `torch_utils.py` **未分发**。公开副本使用新写的 [`rotation_math.py`](training/holomotion/src/utils/rotation_math.py)，覆盖数据加载和 FK 实际需要的数学接口，修改及验证范围见 [`training/README.md`](training/README.md)。保留历史综合声明不表示本包包含那份受限实现或取得其额外授权。

## EngineAI PM01 数值规格

PM01 的关节命名、模型结构相关常数和名义 PD 数值参考 EngineAI Native SDK 与 RL Lab 的对应版本。保留以下两份不同版权主体原文：

- [`EngineAI-Native-SDK-BSD-3-Clause.txt`](LICENSES/EngineAI-Native-SDK-BSD-3-Clause.txt)，来源版本 `335c60e88772c26c7852d0abd6b3c7439037dd8f`。
- [`EngineAI-RL-Lab-BSD-3-Clause.txt`](LICENSES/EngineAI-RL-Lab-BSD-3-Clause.txt)，来源版本 `14ec57be718586bd0ac45375aa1115bd896fbdbc`。

首版仅带离线推理需要的数值规格，不捆绑机器人 MJCF/URDF/mesh、厂商运控二进制或整套厂商安装文件。排除它们是本次发布范围的选择，不表示这些组件一律禁止再分发。完整训练资产须按各自原始来源及条款另行取得。

## 训练数据来源

模型卡和 README 保留了 BONES 要求的署名。八段展示的参考动作均来自 BONES，视频和截图为渲染结果。BONES-SEED 协议第 5.2 节规定结果的权利条件，第 6 节说明限制用途及机器人控制、跟踪模型的适用范围，第 8 节规定署名。结果权利以合规使用、原始数据不可从中提取等条件为前提，原始动作、重定向或镜像数据、数据库描述及逐条素材名称由原来源条款管理。[BONES-SEED 官方协议](https://bones.studio/info/seed-license)

通用池另使用 Carnegie Mellon Graphics Lab Motion Capture Database 的 250 条动作；对应 BVH 转换记录注明 Bruce Hahne，并致谢 NSF EIA-0196217。数据获取及许可应回到 [CMU Motion Capture Database](http://mocap.cs.cmu.edu/) 和实际取得的数据包说明，不能用本仓库代码许可证替代。

新增跑跳参考来自 Ubisoft LAFAN1。[固定来源版本](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/tree/94084601bacdf9cc3764b5c73daaeccae6035fac) 的 [README](https://raw.githubusercontent.com/ubisoft/ubisoft-laforge-animation-dataset/94084601bacdf9cc3764b5c73daaeccae6035fac/README.md) 和 [`license.txt`](https://raw.githubusercontent.com/ubisoft/ubisoft-laforge-animation-dataset/94084601bacdf9cc3764b5c73daaeccae6035fac/license.txt) 标注 **CC BY-NC-ND 4.0**。这不是允许把原动作或改编动作任意转发的许可；本包不分发这些数据。

发布者未取得原数据公开条款之外的额外授权。模型是否涉及 LAFAN1 的受保护表达，以及具体使用或再分发是否需要额外授权，须按模型内容与相应使用方式判断。发布者按其有权授予的范围提供非商业科研许可，CC BY-NC-ND 4.0 的条件按实际适用范围执行。[非商业科研模型许可](models/LICENSE.md) · [Creative Commons 官方 AI 训练指导](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)

## 运行依赖

Torch、NumPy、SciPy、HDF5、ONNX、ONNX Runtime、MuJoCo、MuJoCo-Warp 等依赖不捆绑在此快照中，使用者安装时应保留相应分发条款。历史依赖版本与首版 CPU 检查的环境边界分别记在训练和推理说明中。
