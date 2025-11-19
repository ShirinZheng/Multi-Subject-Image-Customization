import sys
sys.path.append("/content/MultiSubjectGen_Pro")
from src.multi_subject_pipeline import MultiSubjectPipeline
from src.config import Config
import torch

def run_ablation():
    print("\n=== Starting Ablation Study ===")
    print("Goal: Prove that the 'QA Loop' actually improves quality.")
    
    pipe = MultiSubjectPipeline()
    pipe.load_loras()
    
    # --- Experiment A: Without QA Loop  ---
    print("\n[Ablation A] Running WITHOUT QA Loop (Baseline)...")
    # Temporarily save the original configuration
    original_retries = Config.MAX_RETRIES
    # Forcefully disable redrawing to simulate a situation without QA.
    Config.MAX_RETRIES = 0 
    
    img_no_qa = pipe.generate_with_qa_loop()
    img_no_qa.save(f"{Config.PROJECT_ROOT}/output/ablation_no_qa.png")
    print(">> Saved 'ablation_no_qa.png'")
    
    # --- Experiment B: With QA Loop  ---
    print("\n[Ablation B] Running WITH QA Loop (Ours)...")
    # 恢复配置
    Config.MAX_RETRIES = original_retries 
    
    img_with_qa = pipe.generate_with_qa_loop()
    img_with_qa.save(f"{Config.PROJECT_ROOT}/output/ablation_with_qa.png")
    print(">> Saved 'ablation_with_qa.png'")
    
    print("\n Ablation Done! Compare the two images in /output folder.")

if __name__ == "__main__":
    run_ablation()
