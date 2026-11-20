"""Export a compatible 65k Actor checkpoint on CPU; requires the training code."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-code", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    import onnx
    import torch

    sys.path.insert(0, str(args.training_code.resolve()))
    from holomotion.src.algo.supermimic import networks

    torch.set_num_threads(1)
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    run = bundle["specification"]
    config = run["arguments"]
    dims = (run["actor_observation_dim"], run["goal_dim"], config["hist"], config["goal_frames"])
    identity = (bundle["format"], bundle["schedule"]["vector_steps"], config["profile"], config["amp"])
    if identity != ("supermimic_policy_progress_v1", 65000, "paper", "bf16") or dims != (78, 630, 10, 10):
        raise ValueError(f"Unexpected checkpoint contract {identity!r} {dims!r}")
    if set(bundle) != {"format", "pi", "specification", "schedule", "training_wall_s", "resume_supported"}:
        raise ValueError("Expected a lightweight original Actor snapshot")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    names = spec["native_joint_names"]
    if len(names) != 24 or names[-1] != "J23_HEAD_YAW" or spec["policy_action_names"] != names[:23]:
        raise ValueError("Joint contract differs from the training Actor")
    if any(value.dtype != torch.float32 for value in bundle["pi"].values()):
        raise ValueError("The source Actor or its normalizer has unexpected tensor precision")

    # Training uses cuda autocast around enc and dec separately. The portable graph is FP32.
    networks.AMP_DTYPE = None
    policy = networks.Policy(78, 630, hist=10, action_dim=23).cpu().eval()
    policy.load_state_dict(bundle["pi"], strict=True)
    args.models.mkdir(parents=True, exist_ok=True)
    source_copy = args.models / "actor_0000065000.pt"
    onnx_path = args.models / "actor_65000_fp32.onnx"
    meta_path = args.models / "actor_65000.json"
    source_hash = digest(args.checkpoint)
    with tempfile.TemporaryDirectory(dir=args.models) as scratch:
        staging = Path(scratch)
        exported = staging / onnx_path.name
        options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
        with torch.inference_mode():
            torch.onnx.export(
                policy,
                (torch.zeros(1, 10, 78), torch.zeros(1, 630)),
                str(exported),
                input_names=["obs_hist", "goal"], output_names=["action23"],
                opset_version=13, do_constant_folding=True, **options,
            )
        model = onnx.load(str(exported))
        onnx.helper.set_model_props(model, {
            "supermimic_run": "pm01_retrain_011_seed13", "step": "65000",
            "source_actor_sha256": source_hash, "precision": "fp32",
            "observation_order": "oldest_first", "reference_goal": "10x63_stride5",
        })
        onnx.checker.check_model(model)
        onnx.save(model, str(exported))
        metadata = {
            "run": "PM01 SuperMimic retrain 011 seed 13", "vector_step": 65000,
            "source_actor": source_copy.name, "source_actor_sha256": source_hash,
            "onnx": onnx_path.name, "onnx_sha256": digest(exported), "opset": 13,
            "export_robot_spec_sha256": digest(args.spec),
            "source_tensor_dtype": "float32", "source_evaluation_mlp_autocast": "cuda:bfloat16",
            "deployed_forward_dtype": "float32", "onnx_inputs": {"obs_hist": [1,10,78], "goal": [1,630]},
            "onnx_outputs": {"action23": [1,23]}, "history": "oldest to newest; first sample repeated 10 times",
            "observation": ["q - default_q - calibration (24 rad)", "qd (24 rad/s)",
                            "omega in LINK_BASE (3 rad/s)", "unit gravity in LINK_BASE (3)",
                            "previous issued clipped composed action including head (24)"],
            "observation_normalizer": "(obs - obs_norm.mean) / sqrt(obs_norm.var + 0.01**2)",
            "goal_normalizer": "(goal - goal_norm.mean) / sqrt(goal_norm.var + 0.01**2)",
            "goal_frame_offsets": list(range(5, 51, 5)), "future_index_boundary": "repeat last clip frame",
            "goal_per_frame": ["future-current reference position in current reference heading (3)",
                               "future rotation in current reference heading, first 2 columns row-major (6)",
                               "future reference root linear velocity in its own body frame (3)",
                               "future reference root angular velocity in its own body frame (3)",
                               "future reference q (24)", "0.1 * future reference qd (24)"],
            "encoder": [630,2048,1024,512,256,32], "hidden_activation": "ELU",
            "encoder_final": "tanh", "decoder": [812,2048,1024,512,256,23],
            "decoder_input": "concatenate latent32 first, then normalized history flattened",
            "control_hz": 50, "default_calibration_rad": "zero for calibrated model-space joint angles",
            "head": "current reference frame, J23_HEAD_YAW",
            "composed_action": "clip(concat(action23, (head_ref-default_head-calibration_head)/scale_head), -20, 20)",
            "q_target_rad": "default_q + calibration + scale * composed_action",
            "nominal_pd": "clip(kp * (q_target-q) - kd * qd, -effort_limit, effort_limit)",
            "joint_names": names, "network_parameters": sum(v.numel() for v in policy.parameters()),
            "export_torch": str(torch.__version__), "export_onnx": onnx.__version__,
        }
        (staging / meta_path.name).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if args.checkpoint.resolve() != source_copy.resolve():
            shutil.copyfile(args.checkpoint, staging / source_copy.name)
            (staging / source_copy.name).replace(source_copy)
        exported.replace(onnx_path)
        (staging / meta_path.name).replace(meta_path)
    print(f"{onnx_path}  {onnx_path.stat().st_size} bytes, CPU FP32, original Actor at {source_copy}")


if __name__ == "__main__":
    main()
