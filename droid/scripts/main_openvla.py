# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time

import numpy as np
import requests
import json_numpy

json_numpy.patch()

import pandas as pd
from moviepy.editor import ImageSequenceClip
from PIL import Image

from droid.robot_env import RobotEnv
import tqdm
import tyro

faulthandler.enable()

DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters — three cameras (left external, right external, wrist)
    left_camera_id: str = "34520144"
    right_camera_id: str = "37849686"
    wrist_camera_id: str = "15571257"

    # Rollout parameters
    max_timesteps: int = 600
    open_loop_horizon: int = 8
    external_camera: str = "left"
    gripper_mode: str = "binary_hysteresis"  # one of ["continuous", "binary_hysteresis"]
    gripper_open_threshold: float = 0.65
    gripper_close_threshold: float = 0.35
    # If True, swap sent gripper: g -> 1-g after postprocessing (debug / convention check).
    invert_gripper_command: bool = False

    # Remote server parameters (OpenVLA deploy.py)
    # Use localhost for same-machine, or GPU machine IP for cross-machine
    remote_host: str = "localhost"
    remote_port: int = 8000

    # Dataset key for action de-normalization.
    unnorm_key: str | None = "pick_up_red_cube_200"
    run_startup_sanity_check: bool = True
    save_startup_main_camera_image: bool = True
    action_diagnostics_only: bool = False
    action_diagnostics_steps: int = 300
    action_diagnostics_log_every: int = 25


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def _tile_images(left: np.ndarray, wrist: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Tile three camera views into a single image: [left | wrist | right] horizontally."""
    target_h = max(left.shape[0], wrist.shape[0], right.shape[0])
    tiles = []
    for img in (left, wrist, right):
        if img.shape[0] != target_h:
            pil = Image.fromarray(img)
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            pil = pil.resize((new_w, target_h), Image.LANCZOS)
            img = np.array(pil)
        tiles.append(img)
    return np.concatenate(tiles, axis=1)


def main(args: Args):
    assert args.external_camera in {"left", "right"}, (
        f"Please set external_camera to 'left' or 'right', got: {args.external_camera}"
    )
    assert args.gripper_mode in {"continuous", "binary_hysteresis"}, (
        f"Unsupported gripper_mode={args.gripper_mode}. Use 'continuous' or 'binary_hysteresis'."
    )
    assert 0.0 <= args.gripper_close_threshold <= args.gripper_open_threshold <= 1.0, (
        "Expected 0 <= gripper_close_threshold <= gripper_open_threshold <= 1."
    )

    # OpenVLA fine-tuned models output 7-dim Cartesian delta pose: [dx, dy, dz, dRx, dRy, dRz, gripper]
    env = RobotEnv(action_space="cartesian_velocity", gripper_action_space="position")
    print("Created the droid env!")

    server_url = f"http://{args.remote_host}:{args.remote_port}/act"
    print(f"OpenVLA server: {server_url}")

    # One-time startup sanity check to validate request/response contract.
    # This sends a single inference request and prints diagnostics without stepping the robot.
    if args.run_startup_sanity_check:
        print("Running one-time startup sanity check (no robot motion).")
        startup_obs = _extract_observation(args, env.get_observation(), save_to_disk=True)
        startup_payload = _build_payload(args, startup_obs, instruction="pick up the red cube")
        if args.save_startup_main_camera_image:
            Image.fromarray(startup_payload["full_image"]).save("robot_main_camera_view.png")
            print("Saved startup main camera image to robot_main_camera_view.png")
        print(
            "Startup payload diagnostics:",
            {
                "keys": list(startup_payload.keys()),
                "full_image_shape": tuple(startup_payload["full_image"].shape),
                "wrist_image_shape": tuple(startup_payload["wrist_image"].shape),
                "state_shape": tuple(startup_payload["state"].shape),
                "state_dtype": str(startup_payload["state"].dtype),
                "state_min": float(np.min(startup_payload["state"])),
                "state_max": float(np.max(startup_payload["state"])),
                "unnorm_key": startup_payload.get("unnorm_key", None),
            },
        )
        with prevent_keyboard_interrupt():
            startup_response = requests.post(server_url, json=startup_payload)
            startup_response.raise_for_status()
            startup_chunk = np.asarray(startup_response.json(), dtype=np.float32)
        if startup_chunk.ndim == 1:
            startup_chunk = startup_chunk[None, :]
        print(
            "Startup response diagnostics:",
            {
                "chunk_shape": tuple(startup_chunk.shape),
                "chunk_dtype": str(startup_chunk.dtype),
                "first_action": startup_chunk[0].tolist() if startup_chunk.shape[0] > 0 else None,
            },
        )
        if startup_chunk.ndim != 2 or startup_chunk.shape[-1] != 7:
            raise ValueError(f"Expected startup action chunk shape [N, 7], got {startup_chunk.shape}")
        print("Startup sanity check passed.\n")

    if args.action_diagnostics_only:
        _run_action_diagnostics(args, env, server_url, instruction="pick up the red cube")
        return

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    eval_index = 0
    while True:
        # __init__ already homed on first launch; homing again before every subsequent eval.
        if eval_index > 0:
            print("Returning robot to home (reset) before this evaluation...")
            env.reset()
        eval_index += 1

        instruction = "pick up the red cube" #input("Enter instruction: ")
        print(f"Instruction {instruction}")

        video = []
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        prev_gripper_cmd = None

        for t_step in bar:
            start_time = time.time()
            try:
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    save_to_disk=t_step == 0,
                )

                # Tile all three cameras: [left | wrist | right]
                tiled = _tile_images(
                    curr_obs["left_image"],
                    curr_obs["wrist_image"],
                    curr_obs["right_image"],
                )
                video.append(tiled)

                if prev_gripper_cmd is None:
                    prev_gripper_cmd = float(np.clip(curr_obs["gripper_position"], 0.0, 1.0))

                # Query OpenVLA server only when we need a fresh action chunk.
                need_new_chunk = (
                    pred_action_chunk is None
                    or actions_from_chunk_completed >= min(args.open_loop_horizon, pred_action_chunk.shape[0])
                    or actions_from_chunk_completed >= pred_action_chunk.shape[0]
                )
                if need_new_chunk:
                    actions_from_chunk_completed = 0
                    payload = _build_payload(args, curr_obs, instruction)
                    with prevent_keyboard_interrupt():
                        response = requests.post(server_url, json=payload)
                        response.raise_for_status()
                        pred_action_chunk = np.asarray(response.json(), dtype=np.float32)
                    if pred_action_chunk.ndim == 1:
                        pred_action_chunk = pred_action_chunk[None, :]
                    if pred_action_chunk.ndim != 2 or pred_action_chunk.shape[-1] != 7:
                        raise ValueError(f"Expected action chunk shape [N, 7], got {pred_action_chunk.shape}")

                action = pred_action_chunk[actions_from_chunk_completed]
                action_to_env, prev_gripper_cmd = _postprocess_action(args, action, prev_gripper_cmd)
                actions_from_chunk_completed += 1
                env.step(action_to_env)

                elapsed = time.time() - start_time
                if elapsed < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed)
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        save_filename = "video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(
            save_filename + ".mp4", codec="libx264"
        )

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), "
                "or a numeric value 0-100 based on the evaluation spec: "
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0
            else:
                success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be in [0, 100] but got: {success * 100}")
                success = None

        df = df._append(
            {
                "success": success,
                "duration": t_step,
                "video_filename": save_filename,
            },
            ignore_index=True,
        )

        again = input("Do one more eval? (enter y or n) ").lower().strip()
        if again not in ("y", "yes"):
            break

    os.makedirs("results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{ts}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    if left_image is None or right_image is None or wrist_image is None:
        raise RuntimeError("Failed to find left/right/wrist images from observation keys.")

    # Drop alpha channel, convert BGR -> RGB.
    left_image = left_image[..., :3][..., ::-1]
    right_image = right_image[..., :3][..., ::-1]
    wrist_image = wrist_image[..., :3][..., ::-1]

    if save_to_disk:
        combined = np.concatenate([left_image, wrist_image, right_image], axis=1)
        Image.fromarray(combined).save("robot_camera_views.png")

    # DROID RobotEnv exposes proprio under obs_dict["robot_state"], but keep
    # a fallback for wrappers that flatten keys at the top level.
    robot_state = obs_dict.get("robot_state", obs_dict)
    if "cartesian_position" not in robot_state or "gripper_position" not in robot_state:
        raise KeyError(
            "Observation missing proprio keys. Expected cartesian/gripper in "
            "obs_dict['robot_state'] or top-level."
        )

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": np.asarray(robot_state["cartesian_position"], dtype=np.float32),
        "gripper_position": float(robot_state["gripper_position"]),
    }


def _build_payload(args: Args, curr_obs, instruction: str):
    if args.external_camera == "left":
        full_image = curr_obs["left_image"]
    elif args.external_camera == "right":
        full_image = curr_obs["right_image"]
    else:
        raise ValueError(f"Unsupported external_camera={args.external_camera}; use 'left' or 'right'")

    proprio = np.concatenate(
        [
            curr_obs["cartesian_position"],
            [curr_obs["gripper_position"]],
        ],
        axis=0,
    ).astype(np.float32)

    payload = {
        "full_image": full_image.astype(np.uint8),
        "wrist_image": curr_obs["wrist_image"].astype(np.uint8),
        "state": proprio,
        "instruction": instruction,
    }
    if args.unnorm_key is not None:
        payload["unnorm_key"] = args.unnorm_key
    return payload


def _postprocess_action(args: Args, action: np.ndarray, prev_gripper_cmd: float):
    # DROID RobotEnv.step asserts every action dim in [-1, 1]. Gripper position is [0, 1] (open..close), never negative.
    arm_cmd = np.clip(action[:6], -1.0, 1.0)
    raw_gripper = float(action[6])

    if args.gripper_mode == "continuous":
        gripper_cmd = raw_gripper
    else:
        if raw_gripper >= args.gripper_open_threshold:
            gripper_cmd = 1.0
        elif raw_gripper <= args.gripper_close_threshold:
            gripper_cmd = 0.0
        else:
            gripper_cmd = prev_gripper_cmd

    action_to_env = np.concatenate([arm_cmd, [gripper_cmd]], axis=0).astype(np.float32)
    action_to_env[:6] = np.clip(action_to_env[:6], -1.0, 1.0)
    action_to_env[6] = np.clip(action_to_env[6], 0.0, 1.0)
    if args.invert_gripper_command:
        action_to_env[6] = 1.0 - float(action_to_env[6])

    if np.any(np.abs(action_to_env[:6] - action[:6]) > 1e-6) or abs(action_to_env[6] - action[6]) > 1e-6:
        print(
            "Postprocessed action:",
            {"raw": action.tolist(), "processed": action_to_env.tolist(), "gripper_mode": args.gripper_mode},
        )

    return action_to_env, float(action_to_env[6])


def _run_action_diagnostics(args: Args, env: RobotEnv, server_url: str, instruction: str):
    print(
        f"Running action diagnostics only (no robot motion) for {args.action_diagnostics_steps} policy actions."
    )
    pred_action_chunk = None
    chunk_idx = 0
    actions_seen = 0
    all_actions = []

    while actions_seen < args.action_diagnostics_steps:
        step_start = time.time()
        curr_obs = _extract_observation(args, env.get_observation(), save_to_disk=actions_seen == 0)
        need_new_chunk = pred_action_chunk is None or chunk_idx >= pred_action_chunk.shape[0]
        if need_new_chunk:
            payload = _build_payload(args, curr_obs, instruction)
            with prevent_keyboard_interrupt():
                response = requests.post(server_url, json=payload)
                response.raise_for_status()
                pred_action_chunk = np.asarray(response.json(), dtype=np.float32)
            if pred_action_chunk.ndim == 1:
                pred_action_chunk = pred_action_chunk[None, :]
            if pred_action_chunk.ndim != 2 or pred_action_chunk.shape[-1] != 7:
                raise ValueError(f"Expected action chunk shape [N, 7], got {pred_action_chunk.shape}")
            chunk_idx = 0

        action = pred_action_chunk[chunk_idx]
        all_actions.append(action.astype(np.float32))
        chunk_idx += 1
        actions_seen += 1

        if actions_seen % args.action_diagnostics_log_every == 0 or actions_seen == args.action_diagnostics_steps:
            mat = np.stack(all_actions, axis=0)
            per_dim_min = np.min(mat, axis=0)
            per_dim_max = np.max(mat, axis=0)
            oob = np.logical_or(mat < -1.0, mat > 1.0)
            oob_frac = np.mean(oob, axis=0)
            print(
                "Action diagnostics:",
                {
                    "actions_seen": actions_seen,
                    "per_dim_min": per_dim_min.tolist(),
                    "per_dim_max": per_dim_max.tolist(),
                    "oob_fraction_per_dim": oob_frac.tolist(),
                    "total_oob_fraction": float(np.mean(oob)),
                },
            )

        elapsed = time.time() - step_start
        if elapsed < 1 / DROID_CONTROL_FREQUENCY:
            time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed)


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
