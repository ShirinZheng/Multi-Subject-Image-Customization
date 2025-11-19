import matplotlib.pyplot as plt
from PIL import Image
import os
from .config import Config

class Visualizer:
    def __init__(self):
        self.save_dir = Config.PROJECT_ROOT + "/output"
        
    def save_process_grid(self, base_img, masks, final_img, filename="process_viz.png"):

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Mask Preview (Display the two masks overlaid)
        mask_preview = Image.new("RGB", (1024, 1024), "black")
        for m in masks:
            # Make the mask red and semi-transparent overlay.
            colored_mask = Image.new("RGB", (1024, 1024), (255, 50, 50))
            mask_preview = Image.composite(colored_mask, mask_preview, m)
            
        axes[0].imshow(mask_preview)
        axes[0].set_title("1. Spatial Layout / Masks")
        axes[0].axis("off")
        
        # 2. Final Result
        axes[1].imshow(final_img)
        axes[1].set_title("2. Final Composition (Ours)")
        axes[1].axis("off")

        # 3. Detail Zoom (For example, zooming in on a cat's face.，showing Identity Retention)
        # Here's a simple demonstration of cropping the center area.
        width, height = final_img.size
        crop_box = (100, 200, 612, 712) # Approximate location of the cat on the left
        axes[2].imshow(final_img.crop(crop_box)) 
        axes[2].set_title("3. Detail / Identity Check")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f" Process visualization saved to {save_path}")
