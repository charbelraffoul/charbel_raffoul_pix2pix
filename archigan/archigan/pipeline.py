#!/usr/bin/env python3
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

class ArchiPipeline:
    """
    Combine two image folders (A & B) into pix2pix–style AB pairs,
    split into train/val/test, and write to disk.
    """

    def __init__(self, layers, stages):
        """
        Args:
            layers (list of str): [path_to_A_folder, path_to_B_folder]
            stages (list of (int,int)): list of (idx_A, idx_B) pairs
        """
        self.layers = layers
        self.stages = stages

    @staticmethod
    def combineAB(path_A, path_B, path_AB):
        """Read A and B images, concatenate side-by-side, write AB."""
        im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
        im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
        if im_A is None or im_B is None:
            raise IOError(f"Failed to read: {path_A} or {path_B}")
        im_AB = np.concatenate([im_A, im_B], axis=1)
        cv2.imwrite(path_AB, im_AB)

    @staticmethod
    def splits(N, val=0.05, test=0.05, train=0.90):
        """Return three lists of indices for val/test/train splits."""
        idxs = np.random.permutation(N)
        n_val, n_test = int(val * N), int(test * N)
        return [
            idxs[:n_val].tolist(),
            idxs[n_val : n_val + n_test].tolist(),
            idxs[n_val + n_test :].tolist(),
        ]

    def setup_training(self, out_dir):
        """For each stage (A_idx, B_idx), build AB pairs and splits."""
        os.makedirs(out_dir, exist_ok=True)
        for a_idx, b_idx in self.stages:
            dir_A = self.layers[a_idx]
            dir_B = self.layers[b_idx]
            name_A = os.path.basename(dir_A.rstrip(os.sep))
            name_B = os.path.basename(dir_B.rstrip(os.sep))
            pair_name = f"{name_A}__{name_B}"
            dst_root = os.path.join(out_dir, pair_name)
            os.makedirs(dst_root, exist_ok=True)

            fnames = sorted(os.listdir(dir_A))
            val_idxs, test_idxs, train_idxs = self.splits(len(fnames))
            split_map = {
                "val": val_idxs,
                "test": test_idxs,
                "train": train_idxs,
            }

            pbar = tqdm(total=len(fnames), desc=f"Building {pair_name}")
            for split_name, idx_list in split_map.items():
                split_dir = os.path.join(dst_root, split_name)
                os.makedirs(split_dir, exist_ok=True)
                pbar.set_description(f"{pair_name} → {split_name}")
                for i in idx_list:
                    fname = fnames[i]
                    srcA = os.path.join(dir_A, fname)
                    srcB = os.path.join(dir_B, fname)
                    dst  = os.path.join(split_dir, fname)
                    self.combineAB(srcA, srcB, dst)
                    pbar.update(1)
            pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create pix2pix AB dataset from two folders of images"
    )
    parser.add_argument("--A",    required=True, help="Path to folder of A (input) images")
    parser.add_argument("--B",    required=True, help="Path to folder of B (target) images")
    parser.add_argument("--out",  required=True, help="Output directory for AB dataset")
    args = parser.parse_args()

    pipeline = ArchiPipeline(layers=[args.A, args.B], stages=[(0, 1)])
    pipeline.setup_training(args.out)
    print(f"\n✅ Completed: AB dataset written to {args.out}")
