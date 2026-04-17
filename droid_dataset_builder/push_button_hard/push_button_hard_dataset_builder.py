from typing import Iterator, Tuple, Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from PIL import Image

from droid.droid_utils import load_trajectory, crawler
from droid.tfds_utils import MultiThreadedDatasetBuilder


LANGUAGE_INSTRUCTION = "push button hard"
DATA_PATH = "/bluesclues-data/home/pingpong-nima/raj/eccv_26/0_data/push_button_hard_40_eccv"
IMAGE_RES = (720, 1280)


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    def _resize_and_encode(image, size):
        image = Image.fromarray(image)
        return np.array(image.resize(size, resample=Image.BICUBIC))

    def _parse_example(episode_path):
        h5_filepath = os.path.join(episode_path, "trajectory.h5")
        recording_folderpath = os.path.join(episode_path, "recordings", "MP4")

        try:
            data = load_trajectory(h5_filepath, recording_folderpath=recording_folderpath)
        except Exception:
            print(f"Skipping trajectory because data couldn't be loaded for {episode_path}.")
            return None

        lang = LANGUAGE_INSTRUCTION

        try:
            assert all(t.keys() == data[0].keys() for t in data)
            for t in range(len(data)):
                for key in data[0]["observation"]["image"].keys():
                    data[t]["observation"]["image"][key] = _resize_and_encode(
                        data[t]["observation"]["image"][key], (IMAGE_RES[1], IMAGE_RES[0])
                    )

            episode = []
            for i, step in enumerate(data):
                obs = step["observation"]
                action = step["action"]
                camera_type_dict = obs["camera_type"]
                wrist_ids = [k for k, v in camera_type_dict.items() if v == 0]
                exterior_ids = [k for k, v in camera_type_dict.items() if v != 0]

                episode.append(
                    {
                        "observation": {
                            "exterior_image_1_left": obs["image"][f"{exterior_ids[0]}_left"][..., ::-1],
                            "exterior_image_2_left": obs["image"][f"{exterior_ids[1]}_left"][..., ::-1],
                            "wrist_image_left": obs["image"][f"{wrist_ids[0]}_left"][..., ::-1],
                            "cartesian_position": obs["robot_state"]["cartesian_position"],
                            "joint_position": obs["robot_state"]["joint_positions"],
                            "gripper_position": np.array([obs["robot_state"]["gripper_position"]]),
                        },
                        "action_dict": {
                            "cartesian_position": action["cartesian_position"],
                            "cartesian_velocity": action["cartesian_velocity"],
                            "gripper_position": np.array([action["gripper_position"]]),
                            "gripper_velocity": np.array([action["gripper_velocity"]]),
                            "joint_position": action["joint_position"],
                            "joint_velocity": action["joint_velocity"],
                        },
                        "action": np.concatenate((action["cartesian_velocity"], [action["gripper_position"]])),
                        "discount": 1.0,
                        "reward": float((i == (len(data) - 1) and "success" in episode_path)),
                        "is_first": i == 0,
                        "is_last": i == (len(data) - 1),
                        "is_terminal": i == (len(data) - 1),
                        "language_instruction": lang,
                    }
                )
        except Exception:
            print(f"Skipping trajectory because there was an error in data processing for {episode_path}.")
            return None

        sample = {
            "steps": episode,
            "episode_metadata": {
                "file_path": h5_filepath,
                "recording_folderpath": recording_folderpath,
            },
        }
        return episode_path, sample

    for sample in paths:
        yield _parse_example(sample)


class PushButtonHard(MultiThreadedDatasetBuilder):
    name = "push_button_hard"
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    N_WORKERS = 10
    MAX_PATHS_IN_MEMORY = 100
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "exterior_image_1_left": tfds.features.Image(shape=(*IMAGE_RES, 3), dtype=np.uint8, encoding_format="jpeg"),
                                    "exterior_image_2_left": tfds.features.Image(shape=(*IMAGE_RES, 3), dtype=np.uint8, encoding_format="jpeg"),
                                    "wrist_image_left": tfds.features.Image(shape=(*IMAGE_RES, 3), dtype=np.uint8, encoding_format="jpeg"),
                                    "cartesian_position": tfds.features.Tensor(shape=(6,), dtype=np.float64),
                                    "gripper_position": tfds.features.Tensor(shape=(1,), dtype=np.float64),
                                    "joint_position": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                                }
                            ),
                            "action_dict": tfds.features.FeaturesDict(
                                {
                                    "cartesian_position": tfds.features.Tensor(shape=(6,), dtype=np.float64),
                                    "cartesian_velocity": tfds.features.Tensor(shape=(6,), dtype=np.float64),
                                    "gripper_position": tfds.features.Tensor(shape=(1,), dtype=np.float64),
                                    "gripper_velocity": tfds.features.Tensor(shape=(1,), dtype=np.float64),
                                    "joint_position": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                                    "joint_velocity": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "recording_folderpath": tfds.features.Text(),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        print("Crawling all episode paths...")
        episode_paths = crawler(DATA_PATH)
        episode_paths = [
            p for p in episode_paths if os.path.exists(p + "/trajectory.h5") and os.path.exists(p + "/recordings/MP4")
        ]
        print(f"Found {len(episode_paths)} episodes!")
        return {"train": episode_paths}

