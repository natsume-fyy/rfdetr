# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from rfdetr import RFDETRSmall

from visualize_hazydet import generate_representative_visualization


DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR_25"
OUTPUT_DIR = "/root/autodl-tmp/baseline/rf-detr-develop/output/hazydet_small_eca"
VISUALIZE_AFTER_TRAINING = True
FIXED_SAMPLE_FILE = "/root/autodl-tmp/baseline/rf-detr-develop/output/hazydet_fixed_samples_valid.json"


def main() -> None:
    """Train RF-DETR Small on HazyDet and optionally visualize predictions."""
    model = RFDETRSmall()

    model.train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        epochs=8,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        device="cuda",
        num_workers=8,
        use_ema=False,
        checkpoint_interval=5,
        early_stopping=False,
    )

    if VISUALIZE_AFTER_TRAINING:
        generate_representative_visualization(
            dataset_dir=DATASET_DIR,
            checkpoint=f"{OUTPUT_DIR}/checkpoint_best_total.pth",
            output_dir=f"{OUTPUT_DIR}/representative_visualization",
            split="valid",
            confidence_threshold=0.30,
            iou_threshold=0.50,
            sample_file=FIXED_SAMPLE_FILE,
        )


if __name__ == "__main__":
    main()
