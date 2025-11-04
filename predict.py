from cog import BasePredictor, Input
import torch
from PIL import Image
from torchvision import transforms

class Predictor(BasePredictor):
    def setup(self):
        pass  # Här laddar du din tränade modell vid behov

    def predict(
        self,
        image: Input(description="Inputskiss", type=Path),
    ) -> Path:
        return image  # Här returnerar vi bara samma bild – byt mot inference-logik
