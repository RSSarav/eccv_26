import os
from PIL import Image
from droid.droid_utils import load_trajectory

# Your known mapping from the old script
CAM_ID_MAP = {
    "34520144": "rgb_right",
    "15571257": "rgb_wrist",
    "37849686": "rgb_left",
}

episode_path = "/bluesclues-data/home/pingpong-nima/raj/eccv_26/0_data/pick_up_red_cube_200_eccv_new/success/2026-03-18/Wed_Mar_18_05_15_33_2026"
h5_filepath = os.path.join(episode_path, "trajectory.h5")
recording_folderpath = os.path.join(episode_path, "recordings", "MP4")

data = load_trajectory(h5_filepath, recording_folderpath=recording_folderpath)

print(f"Loaded {len(data)} timesteps")

step0 = data[0]
obs = step0["observation"]

print("\n=== camera_type dict ===")
for cam_id, cam_type in obs["camera_type"].items():
    print(f"{cam_id} -> type={cam_type} -> known_name={CAM_ID_MAP.get(str(cam_id), 'UNKNOWN')}")

print("\n=== image keys and shapes ===")
for k, v in sorted(obs["image"].items()):
    print(f"{k}: shape={v.shape}")

# Reproduce exactly what the RLDS builder does
camera_type_dict = obs["camera_type"]
wrist_ids = [str(k) for k, v in camera_type_dict.items() if v == 0]
exterior_ids = [str(k) for k, v in camera_type_dict.items() if v != 0]

print("\n=== RLDS builder assignment ===")
print("wrist_ids    :", wrist_ids)
print("exterior_ids :", exterior_ids)

if len(wrist_ids) > 0:
    print(f"wrist_image_left      <- {wrist_ids[0]}_left ({CAM_ID_MAP.get(wrist_ids[0], 'UNKNOWN')})")
if len(exterior_ids) > 0:
    print(f"exterior_image_1_left <- {exterior_ids[0]}_left ({CAM_ID_MAP.get(exterior_ids[0], 'UNKNOWN')})")
if len(exterior_ids) > 1:
    print(f"exterior_image_2_left <- {exterior_ids[1]}_left ({CAM_ID_MAP.get(exterior_ids[1], 'UNKNOWN')})")

# Save one example image per key so you can visually inspect them
out_dir = "debug_camera_mapping"
os.makedirs(out_dir, exist_ok=True)

print("\n=== saving first-frame debug images ===")
for k, img in obs["image"].items():
    out_path = os.path.join(out_dir, f"{k}.png")
    Image.fromarray(img).save(out_path)
    print(f"saved {out_path}")

print("\nDone.")