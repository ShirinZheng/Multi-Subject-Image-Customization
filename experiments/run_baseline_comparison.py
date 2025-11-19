import sys
sys.path.append("/content/MultiSubjectGen_Pro")
from src.multi_subject_pipeline import MultiSubjectPipeline
from diffusers import StableDiffusionXLPipeline
import torch
from src.config import Config

def run_experiment():
    print("=== Experiment: Baseline vs. Ours ===")
    
    # 1. Baseline: Vanilla SDXL (pure FP32)
    print("Running Baseline (Vanilla SDXL in FP32 Mode)...")
    
    torch.cuda.empty_cache()

    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        Config.MODEL_NAME, 
        use_safetensors=True
    )
    
    base_pipe.enable_model_cpu_offload()
    
    prompt = "a photo of sks cat toy and trk red mug on a table"
    
    # generalizarion
    baseline_img = base_pipe(prompt=prompt, num_inference_steps=30).images[0]
    baseline_img.save(f"{Config.PROJECT_ROOT}/output/baseline_result.png")
    print(" Baseline saved.")
    
    # delete and clean up
    del base_pipe
    torch.cuda.empty_cache()

    # 2. Ours: Multi-Subject Pipeline
    print("Running Ours (Proposed Method)...")
    our_pipe = MultiSubjectPipeline()
    our_pipe.load_loras()
    our_img = our_pipe.generate_with_qa_loop()
    our_img.save(f"{Config.PROJECT_ROOT}/output/ours_result.png")
    print(" Ours saved.")
    
    print("Compare 'baseline_result.png' and 'ours_result.png' in the output folder.")

if __name__ == "__main__":
    run_experiment()
