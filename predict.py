from cog import BasePredictor, Input, Path
from PIL import Image
import torch
from torchvision import transforms
import os

class Predictor(BasePredictor):
    def setup(self):
        # Ladda modellen från en fil
        self.model = torch.load("checkpoints/planritningsmodel.ckpt", map_location="cpu")
        self.model.eval()

        # Bildtransformationer
        self.tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        self.to_pil = transforms.ToPILImage()

    def predict(self, image: Path = Input(description="Input image")) -> Path:
        # Öppna och transformera inputbilden
        input_image = Image.open(str(image)).convert("RGB")
        input_tensor = self.tf(input_image).unsqueeze(0)

        # Kör modellen
        with torch.no_grad():
            output = self.model(input_tensor)[0]

        # Konvertera till PIL-bild och spara
        output_image = self.to_pil(output)
        out_path = "/tmp/output.png"
        output_image.save(out_path)

        return Path(out_path)

