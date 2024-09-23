"""

ostris/ai-toolkit on https://modal.com
Run training with the following command:
modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/modal_lora_config.yaml

"""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import modal
from dotenv import load_dotenv
import re

# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, "/root/ai-toolkit")
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ["DISABLE_TELEMETRY"] = "YES"

# define the volume for storing model outputs, using "creating volumes lazily": https://modal.com/docs/guide/volumes
# you will find your model, samples and optimizer stored in: https://modal.com/storage/your-username/main/flux-lora-models
modal_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)

# modal_output, due to "cannot mount volume on non-empty path" requirement
MOUNT_DIR = "/root/ai-toolkit/modal_output"  # modal_output, due to "cannot mount volume on non-empty path" requirement

# define modal app
image = (
    modal.Image.debian_slim(python_version="3.11")
    # install required system and pip packages, more about this modal approach: https://modal.com/docs/examples/dreambooth_app
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "ruamel.yaml",
        "python-dotenv",
        "torch",
        "diffusers[torch]",
        "transformers",
        "ftfy",
        "torchvision",
        "oyaml",
        "opencv-python",
        "albumentations",
        "safetensors",
        "lycoris-lora==1.8.3",
        "flatten_json",
        "pyyaml",
        "tensorboard",
        "kornia",
        "invisible-watermark",
        "einops",
        "accelerate",
        "toml",
        "pydantic",
        "omegaconf",
        "k-diffusion",
        "open_clip_torch",
        "timm",
        "prodigyopt",
        "controlnet_aux==0.0.7",
        "bitsandbytes",
        "hf_transfer",
        "lpips",
        "pytorch_fid",
        "optimum-quanto",
        "sentencepiece",
        "huggingface_hub",
        "peft",
    )
)


def rename_dataset_files(dataset_path: str) -> None:
    # rename all files as 1.jpg, 2.jpg, 3.jpg, etc (keep the extension same as original)
    import os
    import glob
    import shutil
    from tqdm import tqdm

    files = glob.glob(os.path.join(dataset_path, "*"))
    for i, file in enumerate(tqdm(files)):
        file_name = os.path.basename(file)
        file_ext = os.path.splitext(file_name)[1]
        new_file_name = f"{i+1}{file_ext}"
        shutil.move(file, os.path.join(dataset_path, new_file_name))


def generate_caption(dataset_path: str) -> None:
    import os
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to("cuda")

    # conditional image captioning
    text = "a photograph of"

    # Process each image in the directory
    for filename in os.listdir(dataset_path):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            image_path = os.path.join(dataset_path, filename)

            # Load the image
            image = Image.open(image_path)

            inputs = processor(image, text, return_tensors="pt").to("cuda")

            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            # Save caption to a text file with the same name as the image
            caption_file_path = os.path.splitext(image_path)[0] + ".txt"
            with open(caption_file_path, "w") as caption_file:
                caption_file.write(caption)

            print(
                f"Caption for {filename} saved to {caption_file_path} with content: {caption}"
            )


# mount for the entire ai-toolkit directory
# example: "/Users/username/ai-toolkit" is the local directory, "/root/ai-toolkit" is the remote directory
code_mount = modal.Mount.from_local_dir(
    local_path=os.path.abspath(os.path.join(os.path.dirname(__file__))),
    remote_path="/root/ai-toolkit",
)

# create the Modal app with the necessary mounts and volumes
app = modal.App(
    name="flux-lora-training",
    image=image,
    mounts=[code_mount],
    volumes={MOUNT_DIR: modal_volume},
)

# Check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # Set torch to trace mode
    import torch

    torch.autograd.set_detect_anomaly(True)

import argparse
from toolkit.job import get_job


def print_end_message(jobs_completed, jobs_failed):
    failure_string = (
        f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}"
        if jobs_failed > 0
        else ""
    )
    completed_string = (
        f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"
    )

    print("")
    print("========================================")
    print("Result:")
    if len(completed_string) > 0:
        print(f" - {completed_string}")
    if len(failure_string) > 0:
        print(f" - {failure_string}")
    print("========================================")


@app.function(
    # request a GPU with at least 24GB VRAM
    # more about modal GPU's: https://modal.com/docs/guide/gpu
    gpu="H100",
    # more about modal timeouts: https://modal.com/docs/guide/timeouts
    timeout=7200,  # 2 hours, increase or decrease if needed
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
    ],  # Taking secret from .env file: https://modal.com/docs/guide/secrets
)
def main(config_file_list_str: str, recover: bool = False, name: str = None):
    # convert the config file list from a string to a list
    config_file_list = config_file_list_str.split(",")

    jobs_completed = 0
    jobs_failed = 0

    print(
        f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}"
    )

    for config_file in config_file_list:
        try:
            job = get_job(config_file, name)

            job.config["process"][0]["training_folder"] = MOUNT_DIR
            os.makedirs(MOUNT_DIR, exist_ok=True)
            print(f"Training outputs will be saved to: {MOUNT_DIR}")

            # rename all files as 1.jpg, 2.jpg, 3.jpg, etc (keep the extension same as original)
            rename_dataset_files("/root/ai-toolkit/dataset/")
            # generate caption and store as same name as image file with .txt extension
            generate_caption("/root/ai-toolkit/dataset/")

            # run the job
            job.run()

            # commit the volume after training
            modal_volume.commit()

            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print(f"Error running job: {e}")
            jobs_failed += 1
            if not recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e

    print_end_message(jobs_completed, jobs_failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # require at least one config file
    parser.add_argument(
        "config_file_list",
        nargs="+",
        type=str,
        help="Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially",
    )

    # flag to continue if a job fails
    parser.add_argument(
        "-r",
        "--recover",
        action="store_true",
        help="Continue running additional jobs even if a job fails",
    )

    # optional name replacement for config file
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Name to replace [name] tag in config file, useful for shared config file",
    )
    args = parser.parse_args()

    # convert list of config files to a comma-separated string for Modal compatibility
    config_file_list_str = ",".join(args.config_file_list)

    main.call(
        config_file_list_str=config_file_list_str, recover=args.recover, name=args.name
    )
