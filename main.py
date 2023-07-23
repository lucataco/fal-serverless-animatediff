from fal_serverless import isolated, cached
import time

@cached
def clone_repo():
    import os
    os.system("apt-get update && apt-get install -y git-lfs")
    os.system("git lfs install")
    repo_path = "/data/repos/animatediff/"
    if not os.path.exists(repo_path):
        print("Cloning AnimateDiff repository")
        os.system(f"git clone https://github.com/guoyww/AnimateDiff {repo_path}")
        print("Done")

@cached
def clone_stable_diff():
    import os, sys
    os.chdir('/data/repos/animatediff/')
    sys.path.append('/data/repos/animatediff/')
    repo_path = "models/StableDiffusion/stable-diffusion-v1-5"
    if not os.path.exists(repo_path):
        print("Cloning StableDiffusionv1.5")
        os.system("rm -rf models/StableDiffusion/stable-diffusion-v1-5")
        os.system(f"git clone --branch fp16 https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/stable-diffusion-v1-5")
        print("Done")

@cached
def download_motion_module():
    import os
    model_path = "/data/repos/animatediff/models/Motion_Module/"
    if os.path.exists(model_path):
        print("Downloading model: mm_sd_v14")
        url = "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt"
        os.system(f"cd {model_path} && wget {url}")
        print("Done")

@cached
def download_dreambooth_lora():
    import os
    model_path = "/data/repos/animatediff/models/DreamBooth_LoRA/"
    if os.path.exists(model_path):
        print("Downloading model: toonyou_beta3")
        url = "https://civitai.com/api/download/models/78775"
        os.system(f"cd {model_path} && wget -O toonyou_beta3.safetensors {url}")
        print("Done")

@cached
def model():
    import os, sys
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers.utils.import_utils import is_xformers_available

    os.chdir('/data/repos/animatediff/')
    sys.path.append('/data/repos/animatediff/')
    
    from animatediff.models.unet import UNet3DConditionModel
    from animatediff.pipelines.pipeline_animation import AnimationPipeline

    # Load config
    inference_config_file = "configs/inference/inference.yaml"
    inference_config = OmegaConf.load(inference_config_file)
    pretrained_model_path = "models/StableDiffusion/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path, 
        subfolder="unet", 
        unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
    )

    if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
    
    pipeline = AnimationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
    )
    pipeline = pipeline.to("cuda")
    return pipeline


@isolated(
    kind="conda",
    env_yml="env.yml",
    machine_type="GPU",
    exposed_port=8080,
    keep_alive=300
)
def generate(prompt: str, neg_prompt: str, steps: int, guidance_scale: float):
    import os
    import sys
    import torch
    from safetensors import safe_open

    # Download models
    clone_repo()
    clone_stable_diff()
    download_motion_module()
    download_dreambooth_lora()
    
    # Allow animatediff import and calls
    os.chdir('/data/repos/animatediff/')
    sys.path.append('/data/repos/animatediff/')

    # Import animatediff
    from animatediff.utils.util import save_videos_grid
    from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint
    from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
    
    # Load cached pipeline
    pipeline = model()
    
    # Other params
    lora_alpha=0.8
    base=""
    full_path = "models/DreamBooth_LoRA/toonyou_beta3.safetensors"

    # Load motion model
    motion_path = "models/Motion_Module/mm_sd_v14.ckpt"
    motion_module_state_dict = torch.load(motion_path, map_location="cpu")
    missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
    assert len(unexpected) == 0

    state_dict = {}
    base_state_dict = {}
    with safe_open(full_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    is_lora = all("lora" in k for k in state_dict.keys())
    if not is_lora:
        base_state_dict = state_dict
    else:
        base_state_dict = {}
        with safe_open(base, framework="pt", device="cpu") as f:
            for key in f.keys():
                base_state_dict[key] = f.get_tensor(key)
    # vae
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
    pipeline.vae.load_state_dict(converted_vae_checkpoint)
    # unet
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
    # lora
    if is_lora:
        pipeline = convert_lora(pipeline, state_dict, alpha=lora_alpha)

    pipeline.to("cuda")

    outname = "output.gif"
    outpath = f"./{outname}"
    sample = pipeline(
        prompt,
        negative_prompt     = "",
        num_inference_steps = 25,
        guidance_scale      = 7.5,
        width               = 512,
        height              = 512,
        video_length        = 16,
    ).videos
    samples = torch.concat([sample])
    save_videos_grid(samples, outpath , n_rows=1)
    output_data = None
    with open(outpath, "rb") as file:
        output_data = file.read()

    return output_data


# Main
prompt = "masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, plum blossoms, outdoors, falling petals, white hair, black eyes"
neg_prompt = ""
steps = 25
guidance_scale = 7.5
# Check inference times
t1 = time.time()
gif_data = generate(prompt, neg_prompt, steps, guidance_scale)
t2 = time.time()
print("Inference in: ",t2-t1," seconds")
with open("output.gif", "wb") as f:
    f.write(gif_data)