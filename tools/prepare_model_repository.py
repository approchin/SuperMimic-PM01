#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble the model repository files locally."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit


MODEL_CARD = """---
language:
- en
- zh
license: other
license_name: supermimic-pm01-noncommercial-research
license_link: models/LICENSE.md
tags:
- robotics
- motion-tracking
- reinforcement-learning
- mujoco
- onnx
- supermimic
- pm01
model_name: SuperMimic PM01 checkpoint 65k
---

# SuperMimic PM01 checkpoint 65k

SuperMimic 的 PM01 适配版 v0.1.0，提供单 seed 13 的 65k Actor、FP32 ONNX、
训练与推理源码、历史仿真结果和八段跟踪演示。权重面向非商业科研、教学和评测。

*A PM01 adaptation of SuperMimic with a single-seed checkpoint at 65k collection
steps, an FP32 ONNX export, source code, documented simulation results, and
closed-loop tracking demonstrations for noncommercial research.*
{source_link}

## 架构

训练采用 Actor、世界模型 F 与 Critic；Actor 通过 F 中的想象轨迹优化。
实际推理使用 Actor 和保存的归一化状态。输入为最近 10 帧、每帧 78 维的
本体观测，以及 630 维未来参考；输出 23 维身体动作，再与当前参考头部组合成
24 维关节目标。策略频率为 50 Hz。

详见[架构介绍](docs/architecture.md)与[PM01 适配说明](docs/pm01_adaptation.md)。

## 实验结果

三次固定 200 复测属于原生 MuJoCo-Warp / CUDA BF16 MLP：

| 协议 | 完整片段完成 | 全片归一回报 |
| --- | ---: | ---: |
| 0 ms 复测 A | 195/200 | 0.835604 |
| 0 ms 复测 B | 194/200 | 0.836408 |
| 每动作随机固定 0/5/10/15/20 ms | 194/200 | 0.836666 |

![65k 的原生 BF16 复测](media/figures/native_bf16_retests_65k.png)

这些是发布者报告的历史原生 BF16 仿真结果。固定集参与过模型选择，历史专项含
训练已见参考。本轮只有一个训练 seed。训练最终在 99,918 步因跟踪退化停止，
尚未确认收敛。指标定义、误差、覆盖和证据范围见
[实验介绍](docs/experiments.md)与[详细评测协议](docs/evaluation.md)。

## 跟踪视频

八段 BONES 参考动作的运动画面采用左侧 **Reference**、右侧 **Policy**，以原速
50 fps 播放。参考动作来源为 [Motion Data by Bones Studio](https://bones.studio/)。

[![交叉步舞蹈，左侧 Reference，右侧 Policy](media/videos/showcase/05_cross_steps.jpg)](media/videos/showcase/05_cross_steps.mp4)

视频使用 CPU MuJoCo 3.3.7 与 ONNX Runtime FP32，按动作类型和实际关节及根位置
误差精选。八段均执行至参考末帧，各有至少一项超出原优选条件。视频使用独立于
历史原生 BF16 的仿真协议。持续真机静站、完整敬礼及完整手柄流程尚未验收。
[全部视频与逐段误差](docs/videos.md) · [选片和探索记录](docs/video_selection_v2.md) · [媒体科研使用条款](media/LICENSE.md)

## 4. 权重与离线使用

| 文件 | 内容 |
| --- | --- |
| [`models/actor_0000065000.pt`](models/actor_0000065000.pt) | 轻量 Actor 的元数据脱敏副本 |
| [`models/actor_65000_fp32.onnx`](models/actor_65000_fp32.onnx) | 原 FP32 ONNX，含观测和参考归一化 |
| [`models/SHA256SUMS.json`](models/SHA256SUMS.json) | 文件大小与 SHA-256 |
| `models/actor_tensor_sha256.json` | 原始 26 个模型/归一化张量的内容指纹 |

PT 的私有元数据已清理，全部张量与归档原件逐字节一致；重存改变 PT 文件摘要。
ONNX 保持原文件，其源 PT 摘要指向脱敏前原件。完整说明见[模型卡](docs/model_card.md)。

在独立环境中、模型仓库根目录运行：

```bash
python -m pip install -r inference/requirements.txt
python tools/verify_weights.py
python tools/smoke_inference.py
```

这些命令检查文件与合成输入的 CPU 推理接口。完整训练数据、评测动作集、
机器人资产和完整恢复点不在本模型仓库中，不能据此独立重现整轮训练。

## 许可与署名

权重适用[非商业科研模型许可 v0.1](models/LICENSE.md)。代码采用 [Apache-2.0](LICENSE)，
原始动作数据及第三方材料由各自来源条款管理。[第三方声明](THIRD_PARTY_NOTICES.md)

Training data includes [Motion Data by Bones Studio](https://bones.studio/).

Use of the underlying dataset is subject to the
[BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license).

算法来源为 [SuperMimic](https://arxiv.org/abs/2610.28480)，实现基于
[HoloMotion](https://github.com/HorizonRobotics/HoloMotion) 和固定版本的
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed/tree/a1508d42c9d1699a52e8233ab16d08c488441ab9)。
"""


MODEL_FILES = (
    "models/actor_0000065000.pt",
    "models/actor_65000_fp32.onnx",
    "models/actor_65000.json",
    "models/actor_tensor_sha256.json",
    "models/SHA256SUMS.json",
    "models/LICENSE.md",
)
SHOWCASE_DIR = Path("media/videos/showcase")
SOURCE_DIRECTORIES = ("LICENSES", "inference", "training", "tools", "docs")
PUBLIC_SUFFIXES = {".md", ".txt", ".py", ".json", ".vibed", ".csv", ".png", ".svg"}


def repository_files(root):
    selected = {
        Path(name)
        for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "media/LICENSE.md", *MODEL_FILES)
    }
    for name in (*SOURCE_DIRECTORIES, "media/figures", "media/results"):
        directory = root / name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Expected a release material directory: {name}")
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if path.is_symlink():
                raise ValueError(f"Symlinks are not supported: {relative}")
            if not path.is_file() or any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            if path.suffix in PUBLIC_SUFFIXES or path.name in {"LICENSE", "NOTICE"}:
                selected.add(relative)

    video_manifest = SHOWCASE_DIR / "manifest.json"
    manifest_path = root / video_manifest
    if manifest_path.resolve() != manifest_path or not manifest_path.is_file():
        raise FileNotFoundError(f"Expected the showcase manifest: {video_manifest}")
    selected.add(video_manifest)
    for item in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]:
        for key, suffix in (("video", ".mp4"), ("preview", ".jpg")):
            filename = item[key]
            if Path(filename).name != filename or Path(filename).suffix != suffix:
                raise ValueError(f"Invalid showcase filename: {filename}")
            selected.add(SHOWCASE_DIR / filename)
    for relative in selected:
        path = root / relative
        if path.resolve() != path or not path.is_file():
            raise FileNotFoundError(f"Required release material is missing or linked: {relative}")
    return sorted(selected)


def prepare(output, source_url=None):
    root = Path(__file__).resolve().parents[1]
    output = output.expanduser().resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Choose an output directory outside the source repository")
    if output.exists():
        raise FileExistsError("The output must be a new directory")
    source_link = ""
    if source_url:
        parsed = urlsplit(source_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.query
            or parsed.fragment
            or any(char.isspace() or char in "<>()" for char in source_url)
        ):
            raise ValueError("Use the public HTTPS source repository URL")
        source_link = f"\n[GitHub 源码仓库]({source_url})\n"

    selected = repository_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with tempfile.TemporaryDirectory(prefix=".model-repository-", dir=output.parent) as temporary:
        staging = Path(temporary) / "repository"
        for relative in selected:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
        (staging / "README.md").write_text(MODEL_CARD.format(source_link=source_link), encoding="utf-8")
        for relative in (*selected, Path("README.md")):
            path = staging / relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest[relative.as_posix()] = {"bytes": path.stat().st_size, "sha256": digest}
        record = {"schema_version": 1, "project_version": "0.1.0", "files": manifest}
        (staging / "MODEL_REPOSITORY_MANIFEST.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError("The output must be a new directory")
        staging.rename(output)
    print(json.dumps({"files": len(manifest), "bytes": sum(item["bytes"] for item in manifest.values())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-url", help="Public GitHub source repository URL for the model card")
    args = parser.parse_args()
    prepare(args.output, args.source_url)


if __name__ == "__main__":
    main()
