import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import TestSpecificDataset
from train import PromptIRModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_path",
        default="test/degraded/",
        help="directory of your unlabeled images",
    )
    parser.add_argument(
        "--ckpt",
        default="train_ckpt/epoch=249-step=200000.ckpt",
        help="path to your .ckpt file",
    )
    parser.add_argument(
        "--output_npz", default="pred.npz", help="filename for the output .npz"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda:0", help='e.g. "cuda:0" or "cpu"'
    )
    args = parser.parse_args()

    ds = TestSpecificDataset(args)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PromptIRModel.load_from_checkpoint(
        checkpoint_path=args.ckpt, map_location=args.device
    ).to(args.device)
    model.eval()

    # inference
    outputs = {}
    with torch.no_grad():
        for [name], img in tqdm(loader, desc="Inference"):
            img = img.to(args.device)
            pred = model(img)

            arr = (pred.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

            outputs[f"{name[0]}.png"] = arr[0]

    np.savez(args.output_npz, **outputs)
    print(f"Saved {len(outputs)} images to {args.output_npz}")


if __name__ == "__main__":
    main()
