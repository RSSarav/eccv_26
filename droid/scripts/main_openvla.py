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

    # Remote server parameters (OpenVLA deploy.py)
    # Use localhost for same-machine, or GPU machine IP for cross-machine
    remote_host: str = "localhost"
    remote_port: int = 8000

    # Dataset key for action de-normalization.
    unnorm_key: str | None = "pick_up_red_cube_200"


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
    # OpenVLA fine-tuned models output 7-dim Cartesian delta pose: [dx, dy, dz, dRx, dRy, dRz, gripper]
    env = RobotEnv(action_space="cartesian_velocity", gripper_action_space="position")
    print("Created the droid env!")

    server_url = f"http://{args.remote_host}:{args.remote_port}/act"
    print(f"OpenVLA server: {server_url}")

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = "pick up the red cube" #input("Enter instruction: ")
        print(f"Instruction {instruction}")

        video = []
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")

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

                # Query OpenVLA server for an action chunk.
                with prevent_keyboard_interrupt():
                    response = requests.post(server_url, json=payload)
                    response.raise_for_status()
                    pred_action_chunk = np.asarray(response.json(), dtype=np.float32)

                if pred_action_chunk.ndim == 1:
                    pred_action_chunk = pred_action_chunk[None, :]
                if pred_action_chunk.ndim != 2 or pred_action_chunk.shape[-1] != 7:
                    raise ValueError(f"Expected action chunk shape [N, 7], got {pred_action_chunk.shape}")

                open_loop_horizon = min(args.open_loop_horizon, pred_action_chunk.shape[0])
                for h in range(open_loop_horizon):
                    action = pred_action_chunk[h]
                    action_to_env = np.concatenate([action[:6], [action[6]]], axis=0)
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

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

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

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": np.asarray(obs_dict["cartesian_position"], dtype=np.float32),
        "gripper_position": float(obs_dict["gripper_position"]),
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
