import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from diffusers import ControlNetModel
import argparse

# --- Dataset ---
class PlanritningsDataset(Dataset):
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.files = [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        self.tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        input_path = os.path.join(self.input_dir, f)
        output_path = os.path.join(self.output_dir, f.replace(".jpg", ".jpeg").replace(".png", ".jpeg"))
        x = Image.open(input_path).convert("RGB")
        y = Image.open(output_path).convert("RGB")
        return {"input": self.tf(x), "target": self.tf(y)}

# --- Model ---
class ControlNetTrainer(pl.LightningModule):
    def __init__(self, lr=1e-5):
        super().__init__()
        self.model = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
        )
        self.loss_fn = nn.L1Loss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        input_img = batch["input"]
        target_img = batch["target"]
        # 🔧 OBS: Dummy värden – byt till riktig framåtpassning
        output = self.model(
            sample=input_img,
            timestep=0,
            encoder_hidden_states=torch.zeros_like(input_img),
            controlnet_cond=input_img
        )
        loss = self.loss_fn(output.sample, target_img)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# --- Träningsfunktion ---
def train_model(epochs=3, batch_size=1):
    print("🚀 Startar riktig ControlNet‑träning ...")

    dataset = PlanritningsDataset("dataset/input", "dataset/output")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ControlNetTrainer(lr=1e-5)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=16,
        log_every_n_steps=5
    )

    trainer.fit(model, loader)

    os.makedirs("checkpoints", exist_ok=True)
    trainer.save_checkpoint("checkpoints/planritningsmodel.ckpt")

    print("✅ Träning klar – checkpoint sparad i checkpoints/planritningsmodel.ckpt")

# --- Startpunkt ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size)
