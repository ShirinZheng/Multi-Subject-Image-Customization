from PIL import Image, ImageDraw

class LayoutGenerator:
    def __init__(self, width=1024, height=1024):
        self.width = width
        self.height = height
        
    def get_dual_subject_layout(self):
        base_img = Image.new("RGB", (self.width, self.height), "white")
        
        # Mask 1: left(cat)
        mask1 = Image.new("L", (self.width, self.height), 0)
        draw1 = ImageDraw.Draw(mask1)
        draw1.rectangle([50, 200, 500, 950], fill=255) 
        
        # Mask 2: right(mug)
        mask2 = Image.new("L", (self.width, self.height), 0)
        draw2 = ImageDraw.Draw(mask2)
        draw2.rectangle([524, 200, 980, 950], fill=255)
        
        return base_img, [mask1, mask2]
