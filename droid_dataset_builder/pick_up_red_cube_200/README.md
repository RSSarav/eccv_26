RLDS dataset builder for `pick_up_red_cube_200`.

Source data:
- Raw trajectories and MP4s under `0_data/pick_up_red_cube_200_eccv_new`

Current conversion assumptions:
- One fixed instruction for all episodes: `pick up red cube`
- Wrist camera is selected from `camera_type == 0`
- Two exterior cameras are selected from `camera_type != 0`

