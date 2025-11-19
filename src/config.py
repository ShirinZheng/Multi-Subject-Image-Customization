import os

class Config:
    PROJECT_ROOT = "/content/MultiSubjectGen_Pro"
    MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # training params
    TRAIN_STEPS = 500
    LEARNING_RATE = 1e-4
    
    # QA threshold (automatic feedback mechanism in proposal)
    QA_PASS_THRESHOLD = 23.0 # CLIP Score threshold
    MAX_RETRIES = 2
    
    # Data definition
    SUBJECTS = [
        {
            "name": "cat_toy", 
            "token": "sks", 
            "class": "cat",
            "data": os.path.join(PROJECT_ROOT, "data/cat_toy"),
            "lora_out": os.path.join(PROJECT_ROOT, "checkpoints/lora_cat")
        },
        {
            "name": "red_mug", 
            "token": "trk", 
            "class": "mug",
            "data": os.path.join(PROJECT_ROOT, "data/red_mug"),
            "lora_out": os.path.join(PROJECT_ROOT, "checkpoints/lora_mug")
        }
    ]
