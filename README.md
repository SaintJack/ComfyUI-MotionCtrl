# This is an implementation of MotionCtrl for ComfyUI

[MotionCtrl](https://github.com/TencentARC/MotionCtrl): A Unified and Flexible Motion Controller for Video Generation 

## Install

1. Clone this repo into custom_nodes directory of ComfyUI location

2. Run pip install -r requirements.txt

3. Download the weights of MotionCtrl  [motionctrl.pth](https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl.pth) and put it to `ComfyUI/models/checkpoints`

## Nodes

Core nodes:

- `Load Motionctrl Checkpoint`: load MotionCtrl weights + build sampler
- `Motionctrl Cond`: build conditioning (prompt + camera + traj + mode)
- `Motionctrl Sample Simple`: run DDIM sampling and output IMAGE sequence

Utilities:

- `Load Motion Camera Preset`: load camera pose preset (JSON)
- `Load Motion Traj Preset`: load trajectory preset (TXT/NPY converted to points)
- `Build Motion Camera`: generate camera RT sequence from mode/motion/speed (official semantics)
- `Select Image Indices`: select frames from IMAGE sequence
- `Motionctrl Sample`: advanced sampling node (kept for compatibility)

## Usage

### Quickstart (base workflow)

Open and run:

- `workflow_motionctrl_base.json`

Minimal graph:

1. `Load Motionctrl Checkpoint`
2. `Load Motion Camera Preset` and/or `Load Motion Traj Preset`
3. `Motionctrl Cond`
4. `Motionctrl Sample Simple`
5. `PreviewImage` (or any video combine node)

### Control modes

In `Motionctrl Cond` set `infer_mode`:

- `control camera poses`: camera motion control only
- `control object trajectory`: object motion control only
- `control both camera and object motion`: both

### Image-to-Video (keep the same subject)

If you want the output video to be based on an input image (e.g. a specific cat/flower), connect:

- `LoadImage.IMAGE` -> `Motionctrl Sample Simple.init_image`

Then adjust:

- `keep_init_frames`: lock the first K frames to match the input image in latent space (start with 1~4)

Notes:

- `init_image` is not compatible with `context_overlap` in current implementation (use either one).
- This provides a strong first-frame constraint; subject consistency is still affected by prompt/trajectory strength and model limits.

### Object Motion Control example (flower swaying in the wind)

1. `LoadImage`: choose a flower image
2. `Load Motion Traj Preset`: choose `shake_2` (left-right sway), `frame_length` = 16
3. `Motionctrl Cond`:
   - `infer_mode`: `control object trajectory`
   - `prompt`: `a sunflower swaying in the wind, close-up, natural light, high quality`
4. `Motionctrl Sample Simple`:
   - connect `init_image`
   - set `keep_init_frames` = 1~3

### Custom trajectories / camera poses

- `traj` expects a JSON array of points: `[[x,y], ...]`
  - Supported common scales: `0..1`, `0..256`, `0..1024` (internally normalized)
  - Points are interpolated / uniformly sampled to exactly match `frame_length` (official semantics)
- `camera` expects a JSON array of 12-float lists (flatten 3x4): `[[r11,...,t3], ...]`

### Presets

Camera presets are loaded from `examples/camera_poses/`.
If the repository also contains the official `official/MotionCtrl`, presets from `official/MotionCtrl/dataset/camera_poses/{basic,realestate10k}` will also appear in the dropdown (shown as `basic/...` or `realestate10k/...`).

## Tools

[Motion Traj Tool](https://chaojie.github.io/ComfyUI-MotionCtrl/tools/draw.html) Generate motion trajectories

<img src="assets/traj.png" raw=true>

[Motion Camera Tool](https://chaojie.github.io/ComfyUI-MotionCtrl/tools/index.html) Generate motion camera points

<img src="assets/camera.png" raw=true>

## Examples

base workflow

<img src="assets/base_wf.png" raw=true>

https://github.com/chaojie/ComfyUI-MotionCtrl/blob/main/workflow_motionctrl_base.json

<video controls autoplay="true">
    <source 
   src="assets/dog.mp4" 
   type="video/mp4" 
  />
</video>

unofficial implementation "MotionCtrl deployed on AnimateDiff" workflow:

<img src="assets/scribble_wf.png" raw=true>

https://github.com/chaojie/ComfyUI-MotionCtrl/blob/main/workflow_motionctrl.json

1. Generate LVDM/VideoCrafter Video
2. Select Images->Scribble
3. Use AnimateDiff Scribble SparseCtrl
