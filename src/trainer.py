import subprocess
import os
from .config import Config

def run_training():
    script = "/content/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py"
    for subj in Config.SUBJECTS:
        if os.path.exists(subj['lora_out']):
            print(f"Skipping {subj['name']}, checkpoint exists.")
            continue
            
        print(f"Starting training for {subj['name']}...")
        cmd = [
            "accelerate", "launch", script,
            f"--pretrained_model_name_or_path={Config.MODEL_NAME}",
            f"--instance_data_dir={subj['data']}",
            f"--output_dir={subj['lora_out']}",
            f"--instance_prompt='a photo of {subj['token']} {subj['class']}'",
            "--resolution=1024",
            "--train_batch_size=1",
            "--gradient_accumulation_steps=4",
            f"--learning_rate={Config.LEARNING_RATE}",
            f"--max_train_steps={Config.TRAIN_STEPS}",
            "--mixed_precision=fp16"
        ]
        subprocess.run(" ".join(cmd), shell=True, check=True)
