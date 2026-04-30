import argparse
import datetime
import glob
import json
import math
import os
import tempfile
import folder_paths

import imageio
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path
## note: decord should be imported after torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm
from .lvdm.models.samplers.ddim import DDIMSampler
from .main.evaluation.motionctrl_prompts_camerapose_trajs import (
    both_prompt_camerapose_traj, cmcm_prompt_camerapose, omom_prompt_traj)
from .main.evaluation.motionctrl_inference import motionctrl_sample,save_images,load_camera_pose,load_trajs,load_model_checkpoint,post_prompt,DEFAULT_NEGATIVE_PROMPT
from .utils.utils import instantiate_from_config
from .gradio_utils.traj_utils import process_points,get_flow
from .gradio_utils.camera_utils import CAMERA, process_camera as build_camera_rt
from PIL import Image, ImageFont, ImageDraw
from .gradio_utils.utils import vis_camera
from io import BytesIO

PLUGIN_DIR = Path(__file__).resolve().parent

def _plugin_path(*parts: str) -> str:
    return str(PLUGIN_DIR.joinpath(*parts))

def _find_official_motionctrl_dir() -> Path | None:
    for parent in [PLUGIN_DIR] + list(PLUGIN_DIR.parents)[:6]:
        candidate = parent / "official" / "MotionCtrl"
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None

def _list_camera_preset_options() -> list[str]:
    legacy = [
        "U", "D", "L", "R",
        "O", "O_0.2x", "O_0.4x", "O_1.0x", "O_2.0x",
        "Round-RI", "Round-RI_90", "Round-RI-120", "Round-ZoomIn",
        "SPIN-ACW-60", "SPIN-CW-60",
        "I", "I_0.2x", "I_0.4x", "I_1.0x", "I_2.0x",
        "1424acd0007d40b5", "d971457c81bca597", "018f7907401f2fef", "088b93f15ca8745d", "b133a504fc90a2d1",
    ]
    seen = set()
    out: list[str] = []

    def add(x: str):
        if x not in seen:
            seen.add(x)
            out.append(x)

    for x in legacy:
        add(x)

    examples_dir = Path(_plugin_path("examples", "camera_poses"))
    if examples_dir.exists():
        for p in sorted(examples_dir.glob("test_camera_*.json")):
            stem = p.stem
            suffix = stem[len("test_camera_"):] if stem.startswith("test_camera_") else stem
            add(suffix)

    official_dir = _find_official_motionctrl_dir()
    if official_dir is not None:
        ds_dir = official_dir / "dataset" / "camera_poses"
        for group in ("basic", "realestate10k"):
            gdir = ds_dir / group
            if gdir.exists():
                for p in sorted(gdir.glob("test_camera_*.json")):
                    stem = p.stem
                    suffix = stem[len("test_camera_"):] if stem.startswith("test_camera_") else stem
                    add(f"{group}/{suffix}")

    return out

def _loads_json(value: str, name: str):
    try:
        return json.loads(value)
    except Exception as e:
        raise ValueError(f"{name} 不是合法的 JSON: {e}") from e

def _normalize_points_to_1024(points):
    if not isinstance(points, list) or len(points) == 0:
        raise ValueError("traj 必须是非空的点列表，例如 [[x,y], ...]")
    parsed = []
    max_v = 0.0
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError("traj 点格式错误，必须是 [x,y]")
        x, y = p[0], p[1]
        try:
            x = float(x)
            y = float(y)
        except Exception as e:
            raise ValueError(f"traj 点坐标必须为数字: {e}") from e
        parsed.append([x, y])
        max_v = max(max_v, abs(x), abs(y))
    if max_v <= 1.5:
        scale = 1024.0
    elif max_v <= 256.0 * 1.5:
        scale = 4.0
    else:
        scale = 1.0
    out = [[x * scale, y * scale] for x, y in parsed]
    return out

def process_camera(camera_pose_str,frame_length):
    RT=_loads_json(camera_pose_str, "camera")
    if not isinstance(RT, list) or len(RT) == 0:
        raise ValueError("camera 必须是非空的帧列表，例如 [[...12], ...]")
    for f in RT:
        if not isinstance(f, list) or len(f) != 12:
            raise ValueError("camera 每帧必须是长度为 12 的数组（3x4 展平）")
    for i in range(frame_length):
        if len(RT)<=i:
            RT.append(RT[len(RT)-1])
    
    if len(RT) > frame_length:
        RT = RT[:frame_length]
    
    RT = np.array(RT).reshape(-1, 3, 4)
    return RT


def process_camera_list(camera_pose_str,frame_length):
    RT=_loads_json(camera_pose_str, "camera")
    if not isinstance(RT, list) or len(RT) == 0:
        raise ValueError("camera 必须是非空的帧列表，例如 [[...12], ...]")
    for f in RT:
        if not isinstance(f, list) or len(f) != 12:
            raise ValueError("camera 每帧必须是长度为 12 的数组（3x4 展平）")
    for i in range(frame_length):
        if len(RT)<=i:
            RT.append(RT[len(RT)-1])
    
    if len(RT) > frame_length:
        RT = RT[:frame_length]
        
    RT = np.array(RT).reshape(-1, 3, 4)
    return RT

    
def process_traj(points_str,frame_length):
    points=_loads_json(points_str, "traj")
    points = _normalize_points_to_1024(points)
    points = [[int(x), int(y)] for x, y in points]
    points = process_points(points, frames=frame_length)
    xy_range = 1024
    points = [[int(256*x/xy_range), int(256*y/xy_range)] for x,y in points]
    optical_flow = get_flow(points, video_len=frame_length)
    # optical_flow = torch.tensor(optical_flow).to(device)

    return optical_flow
    
def save_results(video, fps=10,traj="[]",draw_traj_dot=False,cameras=[],draw_camera_dot=False,context_overlap=0):
    
    # b,c,t,h,w
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n)) for framesheet in video] #[3, 1*h, n*w]
    grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [t, h, w*n, 3]
    
    path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

    outframes=[]
    
    #writer = imageio.get_writer(path, format='mp4', mode='I', fps=fps)
    for i in range(grid.shape[0]):
        img = grid[i].numpy()
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        image=Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        #draw.ellipse((0,0,255,255),fill=(255,0,0), outline=(255,0,0))
        if draw_traj_dot:
            traj_list=_normalize_points_to_1024(_loads_json(traj, "traj"))
            
            #print(traj_point)
            size=3
            for j in range(grid.shape[0]):
                traj_point=traj_list[len(traj_list)-1]
                if len(traj_list)>j:
                    traj_point=traj_list[j]
                if i==j:
                    draw.ellipse((traj_point[0]/4-size,traj_point[1]/4-size,traj_point[0]/4+size,traj_point[1]/4+size),fill=(255,0,0), outline=(255,0,0))
                else:
                    draw.ellipse((traj_point[0]/4-size,traj_point[1]/4-size,traj_point[0]/4+size,traj_point[1]/4+size),fill=(255,255,255), outline=(255,255,255))
            
        if draw_camera_dot:
            fig = vis_camera(cameras,1,i)
            camimg=Image.open(BytesIO(fig.to_image('png',256,256)))
            image.paste(camimg,(0,0),camimg.convert('RGBA'))
        
        arr = np.array(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        image_tensor_out = torch.from_numpy(arr.astype(np.float32) / 255.0).contiguous()
        outframes.append(image_tensor_out)
        #writer.append_data(img)

    #writer.close()
    out = torch.stack(outframes[context_overlap:], dim=0)
    if out.ndim == 3:
        out = out.unsqueeze(-1).repeat(1, 1, 1, 3)
    return out

MOTION_TRAJ_OPTIONS = ["curve_1", "curve_2", "curve_3", "curve_4", "horizon_2", "shake_1", "shake_2", "shaking_10"]

        
def read_points(file, video_len=16, reverse=False):
    with open(file, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines:
        x, y = line.strip().split(',')
        points.append((int(x)*4, int(y)*4))
    if reverse:
        points = points[::-1]

    if len(points) > video_len:
        skip = len(points) // video_len
        points = points[::skip]
    points = points[:video_len]
    
    return points
    
class LoadMotionCameraPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_camera": (_list_camera_preset_options(),),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("POINTS",)
    FUNCTION = "load_motion_camera_preset"
    CATEGORY = "motionctrl"
    
    def load_motion_camera_preset(self, motion_camera):
        data="[]"
        preset_path = None
        if "/" in motion_camera:
            group, suffix = motion_camera.split("/", 1)
            official_dir = _find_official_motionctrl_dir()
            if official_dir is not None:
                candidate = official_dir / "dataset" / "camera_poses" / group / f"test_camera_{suffix}.json"
                if candidate.exists():
                    preset_path = str(candidate)
        else:
            candidate = Path(_plugin_path("examples", "camera_poses", f"test_camera_{motion_camera}.json"))
            if candidate.exists():
                preset_path = str(candidate)

        if preset_path is None:
            raise FileNotFoundError(f"找不到相机预设: {motion_camera}")

        with open(preset_path, encoding="utf-8") as f:
            data = f.read()
        
        return (data,)


CAMERA_COMBINE_MODE = [
    "Customized Mode 1: First A then B",
    "Customized Mode 2: Both A and B",
    "Customized Mode 3: RAW Camera Poses",
]

CAMERA_MOTION_OPTIONS = [k for k in CAMERA.keys() if not k.startswith("base_")]

class MotionctrlBuildCamera:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (CAMERA_COMBINE_MODE,),
                "speed": ("FLOAT", {"default": 1.0}),
                "frame_length": ("INT", {"default": 16}),
                "motion_a": (["None"] + CAMERA_MOTION_OPTIONS,),
                "motion_b": (["None"] + CAMERA_MOTION_OPTIONS,),
                "raw_camera_poses": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("POINTS",)
    FUNCTION = "build_camera"
    CATEGORY = "motionctrl"

    def build_camera(self, mode, speed, frame_length, motion_a, motion_b, raw_camera_poses):
        motion_list = []
        if motion_a != "None":
            motion_list.append(motion_a)
        if motion_b != "None":
            motion_list.append(motion_b)

        camera_dict = {
            "speed": float(speed),
            "motion": motion_list,
            "mode": mode,
            "complex": None,
        }
        RT = build_camera_rt(camera_dict, camera_args=raw_camera_poses, num_frames=int(frame_length))
        RT12 = np.array(RT).reshape(-1, 12).tolist()
        return (json.dumps(RT12),)
        

class LoadMotionTrajPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_traj": (MOTION_TRAJ_OPTIONS,),
                "frame_length": ("INT", {"default": 16}),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("POINTS",)
    FUNCTION = "load_motion_traj_preset"
    CATEGORY = "motionctrl"
    
    def load_motion_traj_preset(self, motion_traj, frame_length):
        preset_path = _plugin_path("examples", "trajectories", f"{motion_traj}.txt")
        if not os.path.exists(preset_path):
            raise FileNotFoundError(f"找不到轨迹预设文件: {preset_path}")
        points = read_points(preset_path,frame_length)
        return (json.dumps(points),)

MODE = ["control camera poses", "control object trajectory", "control both camera and object motion"]
class MotionctrlLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "motionctrl.pth"}),
                "frame_length": ("INT", {"default": 16}),
            }
        }
        
    RETURN_TYPES = ("MOTIONCTRL", "EMBEDDER", "VAE", "SAMPLER",)
    RETURN_NAMES = ("model","clip","vae","ddim_sampler",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "motionctrl"

    def load_checkpoint(self, ckpt_name, frame_length):
        gpu_num=1
        gpu_no=0
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        config_path = _plugin_path("configs", "inference", "config_both.yaml")
        args={"ckpt_path":f"{ckpt_path}","adapter_ckpt":None,"base":f"{config_path}","condtype":"both","prompt_dir":None,"n_samples":1,"ddim_steps":50,"ddim_eta":1.0,"bs":1,"height":256,"width":256,"unconditional_guidance_scale":1.0,"unconditional_guidance_scale_temporal":None,"seed":1234,"cond_T":800}
        
        config = OmegaConf.load(args["base"])
        OmegaConf.update(config, "model.params.unet_config.params.temporal_length", frame_length)
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        if not torch.cuda.is_available():
            raise RuntimeError("MotionCtrl 推理需要 CUDA GPU。请使用支持 CUDA 的 PyTorch 环境启动 ComfyUI。")
        model = model.cuda(gpu_no)
        assert os.path.exists(args["ckpt_path"]), f'Error: checkpoint {args["ckpt_path"]} Not Found!'
        print(f'Loading checkpoint from {args["ckpt_path"]}')
        model = load_model_checkpoint(model, args["ckpt_path"], args["adapter_ckpt"])
        model.eval()

        ddim_sampler = DDIMSampler(model)

        return (model,model.cond_stage_model,model.first_stage_model,ddim_sampler,)


class MotionctrlCond:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MOTIONCTRL",),
                "prompt": ("STRING", {"multiline": True, "default":"a rose swaying in the wind"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": DEFAULT_NEGATIVE_PROMPT}),
                "camera": ("STRING", {"multiline": True, "default":"[[1,0,0,0,0,1,0,0,0,0,1,0.2]]"}),
                "traj": ("STRING", {"multiline": True, "default":"[[117, 102]]"}),
                "infer_mode": (MODE, {"default":"control both camera and object motion"}),
                "context_overlap": ("INT", {"default": 0, "min": 0, "max": 32}),
            }
        }
        
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING","TRAJ_LIST","RT_LIST","TRAJ_FEATURES","RT","NOISE_SHAPE","INT")
    RETURN_NAMES = ("positive", "negative","traj_list","rt_list","traj","rt","noise_shape","context_overlap")
    FUNCTION = "load_cond"
    CATEGORY = "motionctrl"

    def load_cond(self, model, prompt, negative_prompt, camera, traj,infer_mode,context_overlap):
        camera_align_file = _plugin_path("camera.json")
        traj_align_file = _plugin_path("traj.json")
        frame_length=model.temporal_length

        camera_align=json.loads(camera)
        for i in range(frame_length):
            if len(camera_align)<=i:
                camera_align.append(camera_align[len(camera_align)-1])
        camera=json.dumps(camera_align)
        traj_align=json.loads(traj)
        for i in range(frame_length):
            if len(traj_align)<=i:
                traj_align.append(traj_align[len(traj_align)-1])
        traj=json.dumps(traj_align)

        if context_overlap>0:
            if os.path.exists(camera_align_file):
                with open(camera_align_file, 'r') as file:
                    pre_camera_align=json.load(file)
                    camera_align=pre_camera_align[:context_overlap]+camera_align[:-context_overlap]

            if os.path.exists(traj_align_file):
                with open(traj_align_file, 'r') as file:
                    pre_traj_align=json.load(file)
                    traj_align=pre_traj_align[:context_overlap]+traj_align[:-context_overlap]

            with open(camera_align_file, 'w') as file:
                json.dump(camera_align, file)

            with open(traj_align_file, 'w') as file:
                json.dump(traj_align, file)
        
        prompts = prompt
        RT = process_camera(camera,frame_length).reshape(-1,12)
        RT_list = process_camera_list(camera,frame_length)
        traj_flow = process_traj(traj,frame_length).transpose(3,0,1,2)
        print(prompts)
        print(RT.shape)
        print(traj_flow.shape)

        height=256
        width=256

        ## run over data
        assert (height % 16 == 0) and (width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        
        ## latent noise shape
        h, w = height // 8, width // 8
        channels = model.channels
        frames = model.temporal_length
        #frames = frame_length
        noise_shape = [1, channels, frames, h, w]

        if infer_mode == MODE[0]:
            camera_poses = RT
            camera_poses = torch.tensor(camera_poses).float()
            camera_poses = camera_poses.unsqueeze(0)
            trajs = None
            if torch.cuda.is_available():
                camera_poses = camera_poses.cuda()
        elif infer_mode == MODE[1]:
            trajs = traj_flow
            trajs = torch.tensor(trajs).float()
            trajs = trajs.unsqueeze(0)
            camera_poses = None
            if torch.cuda.is_available():
                trajs = trajs.cuda()
        else:
            camera_poses = RT
            trajs = traj_flow
            camera_poses = torch.tensor(camera_poses).float()
            trajs = torch.tensor(trajs).float()
            camera_poses = camera_poses.unsqueeze(0)
            trajs = trajs.unsqueeze(0)
            if torch.cuda.is_available():
                camera_poses = camera_poses.cuda()
                trajs = trajs.cuda()
        
        batch_size = noise_shape[0]
        prompts=prompt
        ## get condition embeddings (support single prompt only)
        if isinstance(prompts, str):
            prompts = [prompts]

        for i in range(len(prompts)):
            prompts[i] = f'{prompts[i]}, {post_prompt}'

        cond = model.get_learned_conditioning(prompts)
        if camera_poses is not None:
            RT = camera_poses[..., None]
        else:
            RT = None

        traj_features = None
        if trajs is not None:
            traj_features = model.get_traj_features(trajs)
        else:
            traj_features = None
        
        uc = None
        prompts = batch_size * [negative_prompt]
        uc = model.get_learned_conditioning(prompts)
        if traj_features is not None:
            un_motion = model.get_traj_features(torch.zeros_like(trajs))
        else:
            un_motion = None
        uc = {"features_adapter": un_motion, "uc": uc}

        return (cond,uc,traj,RT_list,traj_features,RT,noise_shape,context_overlap)



class MotionctrlSampleSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MOTIONCTRL",),
                "clip": ("EMBEDDER",),
                "vae": ("VAE",),
                "ddim_sampler": ("SAMPLER",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "traj_list": ("TRAJ_LIST",),
                "rt_list": ("RT_LIST",),
                "traj": ("TRAJ_FEATURES",),
                "rt": ("RT",),
                "steps": ("INT", {"default": 50}),
                "seed": ("INT", {"default": 1234}),
                "eta": ("FLOAT", {"default": 1.0}),
                "guidance_scale": ("FLOAT", {"default": 7.5}),
                "cond_T": ("INT", {"default": 800}),
                "deterministic": ("BOOLEAN", {"default": True}),
                "noise_shape":("NOISE_SHAPE",),
                "context_overlap": ("INT", {"default": 0, "min": 0, "max": 32}),
            },
            "optional": {
                "init_image": ("IMAGE",),
                "keep_init_frames": ("INT", {"default": 1, "min": 0, "max": 32}),
                "traj_tool": ("STRING",{"multiline": False, "default": "https://chaojie.github.io/ComfyUI-MotionCtrl/tools/draw.html"}),
                "draw_traj_dot": ("BOOLEAN", {"default": False}),#, "label_on": "draw", "label_off": "not draw"
                "draw_camera_dot": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_inference"
    CATEGORY = "motionctrl"

    def run_inference(self,model,clip,vae,ddim_sampler,positive, negative,traj_list,rt_list,traj,rt,steps,seed,eta,guidance_scale,cond_T,deterministic,noise_shape,context_overlap,init_image=None,keep_init_frames=1,traj_tool="https://chaojie.github.io/ComfyUI-MotionCtrl/tools/draw.html",draw_traj_dot=False,draw_camera_dot=False):
        frame_length=model.temporal_length
        device = model.betas.device
        print(f'frame_length{frame_length}')
        #noise_shape = [1, 4, 16, 32, 32]
        unconditional_guidance_scale = guidance_scale
        unconditional_guidance_scale_temporal = None
        n_samples = 1
        ddim_steps= steps
        ddim_eta=eta
        #seed = args["seed"]

        if n_samples < 1:
            n_samples = 1
        if n_samples > 4:
            n_samples = 4

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        seed_everything(seed)

        batch_images=[]
        batch_variants = []
        intermediates = {}

        x0=None
        x_T=None
        mask=None
        pre_x0=None
        pre_x_T=None

        pred_x0_path = _plugin_path("pred_x0.pt")
        x_inter_path = _plugin_path("x_inter.pt")
        if context_overlap < 0:
            context_overlap = 0
        if context_overlap >= frame_length:
            context_overlap = frame_length - 1

        if context_overlap == 0:
            if os.path.exists(pred_x0_path):
                os.remove(pred_x0_path)
            if os.path.exists(x_inter_path):
                os.remove(x_inter_path)

        if init_image is not None and keep_init_frames > 0:
            if context_overlap > 0:
                raise ValueError("init_image 与 context_overlap 暂不支持同时启用")
            img0 = init_image[0].detach().cpu().numpy()
            img0 = (img0 * 255.0).clip(0, 255).astype(np.uint8)
            img0 = cv2.resize(img0, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            img0 = img0.astype(np.float32) / 255.0
            img0 = torch.from_numpy(img0).to(device=device, dtype=torch.float32)
            img0 = img0 * 2.0 - 1.0
            img0 = img0.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
            img_video = img0.repeat(1, 1, frame_length, 1, 1)
            x0 = model.encode_first_stage(img_video)
            k = int(keep_init_frames)
            if k >= frame_length:
                k = frame_length
            mask = torch.zeros((1, 1, frame_length, 1, 1), device=device, dtype=torch.float32)
            mask[:, :, :k] = 1.0

        rand_frames = frame_length - context_overlap
        randt=torch.randn([noise_shape[0],noise_shape[1],rand_frames,noise_shape[3],noise_shape[4]], device=device)
        randt_np=randt.detach().cpu().numpy()

        if context_overlap>0:
            if os.path.exists(pred_x0_path):
                pre_x0=torch.load(pred_x0_path, map_location="cpu")
                pre_x0_last = pre_x0[-1] if isinstance(pre_x0, (list, tuple)) else pre_x0
                pre_x0_np=pre_x0_last.detach().cpu().numpy()
                ok = (
                    pre_x0_np.ndim == 5
                    and pre_x0_np.shape[0] == noise_shape[0]
                    and pre_x0_np.shape[1] == noise_shape[1]
                    and pre_x0_np.shape[3] == noise_shape[3]
                    and pre_x0_np.shape[4] == noise_shape[4]
                    and pre_x0_np.shape[2] >= context_overlap
                )
                if ok:
                    pre_x0_np_overlap = np.concatenate((pre_x0_np[:,:,-context_overlap:], randt_np), axis=2)
                    x0=torch.tensor(pre_x0_np_overlap, device=device)
                else:
                    os.remove(pred_x0_path)
            if os.path.exists(x_inter_path):
                pre_x_T=torch.load(x_inter_path, map_location="cpu")
                pre_x_T_last = pre_x_T[-1] if isinstance(pre_x_T, (list, tuple)) else pre_x_T
                pre_x_T_np=pre_x_T_last.detach().cpu().numpy()
                ok = (
                    pre_x_T_np.ndim == 5
                    and pre_x_T_np.shape[0] == noise_shape[0]
                    and pre_x_T_np.shape[1] == noise_shape[1]
                    and pre_x_T_np.shape[3] == noise_shape[3]
                    and pre_x_T_np.shape[4] == noise_shape[4]
                    and pre_x_T_np.shape[2] >= context_overlap
                )
                if ok:
                    pre_x_T_np_overlap = np.concatenate((pre_x_T_np[:,:,-context_overlap:], randt_np), axis=2)
                    x_T=torch.tensor(pre_x_T_np_overlap, device=device)
                else:
                    os.remove(x_inter_path)
        
        for _ in range(n_samples):
            if ddim_sampler is not None:
                uc = None if unconditional_guidance_scale == 1.0 else negative
                samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                                conditioning=positive,
                                                batch_size=noise_shape[0],
                                                shape=noise_shape[1:],
                                                verbose=False,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta,
                                                temporal_length=noise_shape[2],
                                                conditional_guidance_scale_temporal=unconditional_guidance_scale_temporal,
                                                features_adapter=traj,
                                                pose_emb=rt,
                                                cond_T=cond_T,
                                                mask=mask,
                                                x0=x0,
                                                x_T=x_T
                                                )        
            #print(f'{samples}')
            ## reconstruct from latent to pixel space
            batch_images = model.decode_first_stage(samples)
            batch_variants.append(batch_images)
            '''
            batch_images = model.decode_first_stage(intermediates['pred_x0'][0])
            batch_variants.append(batch_images)
            batch_images = model.decode_first_stage(intermediates['pred_x0'][1])
            batch_variants.append(batch_images)
            batch_images = model.decode_first_stage(intermediates['pred_x0'][2])
            batch_variants.append(batch_images)
            batch_images = model.decode_first_stage(intermediates['x_inter'][0])
            batch_variants.append(batch_images)
            batch_images = model.decode_first_stage(intermediates['x_inter'][1])
            batch_variants.append(batch_images)
            batch_images = model.decode_first_stage(intermediates['x_inter'][2])
            batch_variants.append(batch_images)
            '''
        ## variants, batch, c, t, h, w
        batch_variants = torch.stack(batch_variants, dim=1)
        batch_variants = batch_variants[0]
        
        torch.save(intermediates['x_inter'], x_inter_path)
        torch.save(intermediates['pred_x0'], pred_x0_path)
        ret = save_results(batch_variants, fps=10,traj=traj_list,draw_traj_dot=draw_traj_dot,cameras=rt_list,draw_camera_dot=draw_camera_dot,context_overlap=context_overlap)
        #print(ret)
        return ret
        

class MotionctrlSample:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default":"a rose swaying in the wind"}),
                "camera": ("STRING", {"multiline": True, "default":"[[1,0,0,0,0,1,0,0,0,0,1,0.2]]"}),
                "traj": ("STRING", {"multiline": True, "default":"[[117, 102]]"}),
                "frame_length": ("INT", {"default": 16}),
                "steps": ("INT", {"default": 50}),
                "seed": ("INT", {"default": 1234}),
            },
            "optional": {
                "traj_tool": ("STRING",{"multiline": False, "default": "https://chaojie.github.io/ComfyUI-MotionCtrl/tools/draw.html"}),
                "draw_traj_dot": ("BOOLEAN", {"default": False}),#, "label_on": "draw", "label_off": "not draw"
                "draw_camera_dot": ("BOOLEAN", {"default": False}),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "motionctrl.pth"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_inference"
    CATEGORY = "motionctrl"
        
    def run_inference(self,prompt,camera,traj,frame_length,steps,seed,traj_tool="https://chaojie.github.io/ComfyUI-MotionCtrl/tools/draw.html",draw_traj_dot=False,draw_camera_dot=False,ckpt_name="motionctrl.pth"):
        gpu_num=1
        gpu_no=0
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        config_path = _plugin_path("configs", "inference", "config_both.yaml")
        args={"savedir":f'./output/both_seed20230211',"ckpt_path":f"{ckpt_path}","adapter_ckpt":None,"base":f"{config_path}","condtype":"both","prompt_dir":None,"n_samples":1,"ddim_steps":50,"ddim_eta":1.0,"bs":1,"height":256,"width":256,"unconditional_guidance_scale":1.0,"unconditional_guidance_scale_temporal":None,"seed":1234,"cond_T":800,"save_imgs":True,"cond_dir":"./custom_nodes/ComfyUI-MotionCtrl/examples/"}
        args["cond_dir"] = _plugin_path("examples")
        
        prompts = prompt
        RT = process_camera(camera,frame_length).reshape(-1,12)
        RT_list = process_camera_list(camera,frame_length)
        traj_flow = process_traj(traj,frame_length).transpose(3,0,1,2)
        print(prompts)
        print(RT.shape)
        print(traj_flow.shape)
        
        args["savedir"]=f'./output/{args["condtype"]}_seed{args["seed"]}'
        config = OmegaConf.load(args["base"])
        OmegaConf.update(config, "model.params.unet_config.params.temporal_length", frame_length)
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        model = model.cuda(gpu_no)
        assert os.path.exists(args["ckpt_path"]), f'Error: checkpoint {args["ckpt_path"]} Not Found!'
        print(f'Loading checkpoint from {args["ckpt_path"]}')
        model = load_model_checkpoint(model, args["ckpt_path"], args["adapter_ckpt"])
        model.eval()
       
        ## run over data
        assert (args["height"] % 16 == 0) and (args["width"] % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        
        ## latent noise shape
        h, w = args["height"] // 8, args["width"] // 8
        channels = model.channels
        frames = model.temporal_length
        #frames = frame_length
        noise_shape = [args["bs"], channels, frames, h, w]

        savedir = os.path.join(args["savedir"], "samples")
        os.makedirs(savedir, exist_ok=True)
        
        #noise_shape = [1, 4, 16, 32, 32]
        unconditional_guidance_scale = 7.5
        unconditional_guidance_scale_temporal = None
        n_samples = 1
        ddim_steps= steps
        ddim_eta=1.0
        cond_T=800
        #seed = args["seed"]

        if n_samples < 1:
            n_samples = 1
        if n_samples > 4:
            n_samples = 4

        seed_everything(seed)
        
        camera_poses = RT
        trajs = traj_flow
        camera_poses = torch.tensor(camera_poses).float()
        trajs = torch.tensor(trajs).float()
        camera_poses = camera_poses.unsqueeze(0)
        trajs = trajs.unsqueeze(0)
        if torch.cuda.is_available():
            camera_poses = camera_poses.cuda()
            trajs = trajs.cuda()
        
        ddim_sampler = DDIMSampler(model)
        batch_size = noise_shape[0]
        prompts=prompt
        ## get condition embeddings (support single prompt only)
        if isinstance(prompts, str):
            prompts = [prompts]

        for i in range(len(prompts)):
            prompts[i] = f'{prompts[i]}, {post_prompt}'

        cond = model.get_learned_conditioning(prompts)
        if camera_poses is not None:
            RT = camera_poses[..., None]
        else:
            RT = None

        traj_features = None
        if trajs is not None:
            traj_features = model.get_traj_features(trajs)
        else:
            traj_features = None
            
        uc = None
        if unconditional_guidance_scale != 1.0:
            # prompts = batch_size * [""]
            prompts = batch_size * [DEFAULT_NEGATIVE_PROMPT]
            uc = model.get_learned_conditioning(prompts)
            if traj_features is not None:
                un_motion = model.get_traj_features(torch.zeros_like(trajs))
            else:
                un_motion = None
            uc = {"features_adapter": un_motion, "uc": uc}
        else:
            uc = None
        
        batch_images=[]
        batch_variants = []
        for _ in range(n_samples):
            if ddim_sampler is not None:
                samples, _ = ddim_sampler.sample(S=ddim_steps,
                                                conditioning=cond,
                                                batch_size=noise_shape[0],
                                                shape=noise_shape[1:],
                                                verbose=False,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta,
                                                temporal_length=noise_shape[2],
                                                conditional_guidance_scale_temporal=unconditional_guidance_scale_temporal,
                                                features_adapter=traj_features,
                                                pose_emb=RT,
                                                cond_T=cond_T
                                                )        
            #print(f'{samples}')
            ## reconstruct from latent to pixel space
            batch_images = model.decode_first_stage(samples)
            batch_variants.append(batch_images)
        ## variants, batch, c, t, h, w
        batch_variants = torch.stack(batch_variants, dim=1)
        batch_variants = batch_variants[0]
        
        ret = save_results(batch_variants, fps=10,traj=traj,draw_traj_dot=draw_traj_dot,cameras=RT_list,draw_camera_dot=draw_camera_dot)
        #print(ret)
        return ret
        
        
class ImageSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "selected_indexes": ("STRING", {
                    "multiline": False,
                    "default": "1,2,3"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "run"

    OUTPUT_NODE = False

    CATEGORY = "motionctrl"

    def run(self, images: torch.Tensor, selected_indexes: str):
        shape = images.shape
        len_first_dim = shape[0]

        selected_index: list[int] = []
        total_indexes: list[int] = list(range(len_first_dim))
        for s in selected_indexes.strip().split(','):
            try:
                if ":" in s:
                    _li = s.strip().split(':', maxsplit=1)
                    _start = _li[0]
                    _end = _li[1]
                    if _start and _end:
                        selected_index.extend(
                            total_indexes[int(_start):int(_end)]
                        )
                    elif _start:
                        selected_index.extend(
                            total_indexes[int(_start):]
                        )
                    elif _end:
                        selected_index.extend(
                            total_indexes[:int(_end)]
                        )
                else:
                    x: int = int(s.strip())
                    if x < len_first_dim:
                        selected_index.append(x)
            except:
                pass

        if selected_index:
            print(f"ImageSelector: selected: {len(selected_index)} images")
            return (images[selected_index, :, :, :], )

        print(f"ImageSelector: selected no images, passthrough")
        return (images, )


NODE_CLASS_MAPPINGS = {
    "Motionctrl Sample":MotionctrlSample,
    "Motionctrl Sample Simple":MotionctrlSampleSimple,
    "Load Motion Camera Preset":LoadMotionCameraPreset,
    "Load Motion Traj Preset":LoadMotionTrajPreset,
    "Build Motion Camera": MotionctrlBuildCamera,
    "Select Image Indices": ImageSelector,
    "Load Motionctrl Checkpoint": MotionctrlLoader,
    "Motionctrl Cond": MotionctrlCond,
}
