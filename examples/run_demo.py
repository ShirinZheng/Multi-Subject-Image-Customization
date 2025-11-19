import sys
sys.path.append("/content/MultiSubjectGen_Pro")
from src.trainer import run_training
from experiments.run_baseline_comparison import run_experiment
from experiments.run_ablation import run_ablation
from src.visualization import Visualizer
from PIL import Image
from src.config import Config
from src.spatial_layout import LayoutGenerator

def main():
    print(" Starting Full Project Demo ...")
    
    # 1. Training check (skip if a model already exists)
    run_training()
    
    # 2. run Baseline comparation (Vanilla SDXL vs Ours)
    # This will generate baseline_result.png and ours_result.png
    run_experiment()
    
    # 3. Running ablation experiments (No-QA vs. With-QA)
    # This will generate ablation_no_qa.png and ablation_with_qa.png
    run_ablation()
    
    # 4. Generate visual analysis charts.
    print("\n Generating Visual Analysis...")
    viz = Visualizer()
    
    # Load the "Ours" result that was just generated.
    try:
        final_img = Image.open(f"{Config.PROJECT_ROOT}/output/ours_result.png")
        # Reacquire Mask only for drawing.
        layout_gen = LayoutGenerator()
        _, masks = layout_gen.get_dual_subject_layout()
        
        viz.save_process_grid(None, masks, final_img, filename="final_report_viz.png")
    except Exception as e:
        print(f" Skipped visualization: {e}")
    
    print("\n Demo Finished! All results are in /content/MultiSubjectGen_Pro/output")

if __name__ == "__main__":
    main()
