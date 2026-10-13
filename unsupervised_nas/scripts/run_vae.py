#!/usr/bin/env python3
"""VAE基线"""
import argparse
import torch
from pathlib import Path
from baselines.vae import VAEBaseline
from datasets import MVTecWoodDataset, MVTecTestDataset
from torchvision import transforms
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../datasets/mvtec_anomaly_detection/wood')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default='experiments/baselines/vae')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = MVTecWoodDataset(root=args.data_dir, split='train', defect_type='good', transform=transform)
    test_dataset = MVTecTestDataset(root=args.data_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = VAEBaseline(latent_dim=128).to(device)
    model.train(train_loader, epochs=args.epochs, lr=args.lr)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save(Path(args.output_dir) / 'vae_model.pth')
    
    metrics = model.evaluate(test_loader)
    print(f"\nVAE Results:")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  AP: {metrics['ap']:.4f}")
    
    import json
    with open(Path(args.output_dir) / 'results.json', 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
