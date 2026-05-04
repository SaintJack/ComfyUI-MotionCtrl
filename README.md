# This is an implementation of MotionCtrl for ComfyUI

[MotionCtrl](https://github.com/TencentARC/MotionCtrl): A Unified and Flexible Motion Controller for Video Generation 

## Install

### Install into ComfyUI (custom_nodes)

1. Locate your ComfyUI folder (the one containing `main.py`).

2. Copy / clone this folder into `ComfyUI/custom_nodes/` so the final path looks like:

- `ComfyUI/custom_nodes/ComfyUI-MotionCtrl/`

Example (git):

```bash
cd <ComfyUI>
cd custom_nodes
git clone <this-repo-url> ComfyUI-MotionCtrl
```

3. Install Python dependencies in the same Python environment ComfyUI uses:

```bash
cd ComfyUI/custom_nodes/ComfyUI-MotionCtrl
pip install -r requirements.txt
```

4. Download MotionCtrl weight `motionctrl.pth` and put it to:

- `ComfyUI/models/checkpoints/motionctrl.pth`

5. Restart ComfyUI. The nodes will appear under category `motionctrl`.

### Update from a previous version

If you already have an older `ComfyUI-MotionCtrl` installed:

1. Stop ComfyUI.
2. Go to the plugin folder:

```bash
cd <ComfyUI>/custom_nodes/ComfyUI-MotionCtrl
```

3. Update the code:

- If you installed via git:

```bash
git pull
```

- If you installed by copying files: replace the whole `ComfyUI/custom_nodes/ComfyUI-MotionCtrl/` folder with the new version.

4. Re-install dependencies (safe to run again):

```bash
pip install -r requirements.txt
```

5. Restart ComfyUI.

Notes:

- If ComfyUI still shows old nodes, delete `__pycache__` under `ComfyUI/custom_nodes/ComfyUI-MotionCtrl/` and restart.
- Workflows saved with older node signatures may need to be reloaded after updating.

### Notes (Windows)

- If you run ComfyUI via a bundled/portable Python, run `pip` from that Python (so dependencies are installed to the correct environment).
- If ComfyUI fails to import a dependency, check the console log at startup; it usually indicates which package is missing.

### Minimal steps (summary)

1. Put this repo under `ComfyUI/custom_nodes/ComfyUI-MotionCtrl/`
2. `pip install -r requirements.txt`
3. Put `motionctrl.pth` under `ComfyUI/models/checkpoints/`
4. Restart ComfyUI

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

SVD (Image-to-Video) nodes:

- `Load Motionctrl+SVD Checkpoint`: load MotionCtrl+SVD checkpoint (`motionctrl_svd.ckpt`) and build the SVD pipeline
- `Motionctrl+SVD Sample`: image-to-video sampling with camera pose conditioning (RT) and SVD motion controls (`fps_id`, `motion_bucket_id`)

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

### MotionCtrl + SVD (Image-to-Video)

This mode uses the official MotionCtrl `svd` branch pipeline (image-to-video), which is much better at preserving the input subject identity.

Requirements:

- Put `motionctrl_svd.ckpt` under `ComfyUI/models/checkpoints/` (same folder as other checkpoints).
- The SVD pipeline code must be available on disk. This repo supports auto-discovery if you have:
  - `official/MotionCtrl_svd/` (the MotionCtrl repository cloned with `--branch svd`)
  - Or set `svd_repo_path` explicitly in `Load Motionctrl+SVD Checkpoint`.
- Install the extra Python dependencies required by the SVD pipeline (refer to `official/MotionCtrl_svd/requirements.txt`).

Workflow:

- `workflow_motionctrl_svd_i2v_camera.json`

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
