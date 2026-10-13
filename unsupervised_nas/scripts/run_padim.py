#!/usr/bin/env python3
"""PaDiM基线"""
import argparse, torch, json
from pathlib import Path
from baselines.padim import PaDiMModel
from datasets import MVTecWoodDataset, MVTecTestDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../datasets/mvtec_anomaly_detection/wood')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default='experiments/baselines/padim')
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = PaDiMModel(backbone=args.backbone).to(device)
    model.train(train_loader)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save(Path(args.output_dir) / 'padim_model.pkl')
    
    metrics = model.evaluate(test_loader)
    print(f"\nPaDiM Results:\n  AUROC: {metrics['auroc']:.4f}\n  AP: {metrics['ap']:.4f}")
    
    with open(Path(args.output_dir) / 'results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()
