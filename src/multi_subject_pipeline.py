import torch
from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL, EulerDiscreteScheduler
from .config import Config
from .evaluation import Evaluator
from .spatial_layout import LayoutGenerator
from PIL import Image, ImageFilter

class MultiSubjectPipeline:
    def __init__(self, device="cuda"):
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(device)
        
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            Config.MODEL_NAME, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
        self.layout_gen = LayoutGenerator()
        self.evaluator = Evaluator()
        self.device = device
        
    def load_loras(self):
        print(">>> Loading LoRAs...")
        for subj in Config.SUBJECTS:
            try:
                self.pipe.load_lora_weights(subj['lora_out'], weight_name="pytorch_lora_weights.safetensors", adapter_name=subj['name'])
            except: pass
            
    def generate_with_qa_loop(self):
        base_img, masks = self.layout_gen.get_dual_subject_layout()
        masks = [m.filter(ImageFilter.GaussianBlur(radius=30)) for m in masks]
        
        print("--- Phase 1: Generating Coherent Global Scene ---")

        self.pipe.disable_lora()
        
        global_prompt = "a wide shot of a cat sitting on the left and a red mug on the right on a continuous dark walnut wooden table, sunlit living room, bokeh background, highly detailed, 4k, photorealistic"
        
        empty_bg = Image.new("RGB", (1024, 1024), "gray")
        full_mask = Image.new("L", (1024, 1024), 255)
        
        current_img = self.pipe(
            prompt=global_prompt,
            negative_prompt="split view, collage, watermark, text, drawing",
            image=empty_bg, mask_image=full_mask,
            num_inference_steps=30, strength=1.0, guidance_scale=7.5
        ).images[0]
        
        print(">>> Global scene generated. Now injecting identities...")
        
        # Phase 2: Identity Injection

        self.pipe.enable_lora()
        
        for i, subj in enumerate(Config.SUBJECTS):
            print(f"--- Injecting {subj['name']} into scene ---")
            
            self.pipe.set_adapters([subj['name']], adapter_weights=[0.9])
            
            prompt = f"a photo of {subj['token']} {subj['class']}, sitting on a dark walnut wooden table, realistic"
            

            current_img = self.pipe(
                prompt=prompt,
                negative_prompt="blur, bad anatomy, ghost, transparent",
                image=current_img,  
                mask_image=masks[i],  
                num_inference_steps=35,
                strength=0.75,         
                guidance_scale=7.5
            ).images[0]

        # Phase 3: QA Loop
        print(">>> Entering QA Loop...")
        for i, subj in enumerate(Config.SUBJECTS):
            prompt = f"a photo of {subj['token']} {subj['class']}"
            score = self.evaluator.compute_clip_score(current_img, prompt)
            
            # threshold 23.0
            if score < Config.QA_PASS_THRESHOLD:
                print(f"!!! Triggering Refinement for {subj['name']} (Score: {score:.2f})")
                self.pipe.set_adapters([subj['name']], adapter_weights=[1.0])
                current_img = self.pipe(
                    prompt=prompt + ", masterpiece, high fidelity", 
                    image=current_img, mask_image=masks[i], 
                    num_inference_steps=40, strength=0.65
                ).images[0]
            else:
                print(f"Subject {subj['name']} Passed (Score: {score:.2f})")
                
        return current_img
