import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageTextDataset(Dataset):
    def __init__(self, json_file, img_dir, processor, max_length=77):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = read_image(os.path.join(self.img_dir, item['image']))
        caption = item['caption']

        inputs = self.processor(
            text=[caption],
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )

        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        }

def train(args):
    # Load pre-trained model and processor
    model = CLIPModel.from_pretrained(args.model_name)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    # Prepare dataset and dataloader
    dataset = ImageTextDataset(args.json_file, args.img_dir, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(dataloader) * args.num_epochs
    )

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    model.to(args.device)
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        # Gradual unfreezing
        if epoch < args.freeze_epochs:
            for param in model.text_model.parameters():
                param.requires_grad = False
            for param in model.vision_model.parameters():
                param.requires_grad = False
        elif epoch == args.freeze_epochs:
            for param in model.text_model.parameters():
                param.requires_grad = True
            for param in model.vision_model.parameters():
                param.requires_grad = True

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"):
            pixel_values = batch['pixel_values'].to(args.device)
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    # Save final model
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Finetune CLIP model")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="Name or path of pre-trained CLIP model")
    parser.add_argument("--json_file", type=str, required=True, help="Path to JSON file containing image-caption pairs")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save finetuned model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every n epochs")
    parser.add_argument("--freeze_epochs", type=int, default=1, help="Number of epochs to freeze pre-trained weights")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)

if __name__ == "__main__":
    main()