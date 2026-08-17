import argparse
import glob
import os
import re
import sys
import uuid

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rapidlidar.models.rapidlidar import RapidLiDAR


def main(args):
    torch.set_float32_matmul_precision("medium")

    model = RapidLiDAR(
        config_path=args.config_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_reconstruction_rounds=args.num_reconstruction_rounds,
        predict_residual=args.predict_residual,
        init_noise_std=args.init_noise_std,
        displacement_scale=args.displacement_scale,
        voxel_size=args.voxel_size,
    )

    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    if args.test:
        logger = None
        run_name = "test"
        run_id = "test"
    else:
        run_id = slurm_job_id if slurm_job_id else uuid.uuid4().hex[:8]
        logger = WandbLogger(project=args.experiment, id=run_id, resume="allow")
        run_name = logger.experiment.name if logger.experiment else run_id
        with open(args.config_path, "r") as f:
            logger.log_hyperparams({"config": f.read()})

    run_ckpt_dir = f"checkpoints/{args.experiment}/{run_id}"
    print("Checkpoints directory:", run_ckpt_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_cd", dirpath=run_ckpt_dir, filename=f"{run_name}_best",
        save_top_k=1, mode="min", save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"{run_name}_last"
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [] if args.test else [checkpoint_callback, lr_monitor]

    devices = -1 if args.device == -1 else [args.device]

    trainer = pl.Trainer(
        default_root_dir=run_ckpt_dir,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=devices,
        strategy="ddp_find_unused_parameters_false",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        enable_checkpointing=not args.test,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        precision=args.precision,
    )

    if args.test:
        trainer.test(model, ckpt_path=args.ckpt_path)
    else:
        hpc_ckpts = sorted(
            glob.glob(os.path.join(run_ckpt_dir, "hpc_ckpt_*.ckpt")),
            key=lambda p: int(re.search(r"hpc_ckpt_(\d+)", p).group(1)),
        )
        resume_ckpt = hpc_ckpts[-1] if hpc_ckpts else args.resume_ckpt
        trainer.fit(model, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/rapidlidar.yaml")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--experiment", "-e", type=str, default="rapidlidar")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--limit_test_batches", type=int, default=None)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", "-d", type=int, default=-1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--num_reconstruction_rounds", type=int, default=2)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    _pr = parser.add_mutually_exclusive_group()
    _pr.add_argument("--predict_residual", dest="predict_residual", action="store_true", default=True)
    _pr.add_argument("--no_predict_residual", dest="predict_residual", action="store_false")
    parser.add_argument("--init_noise_std", type=float, default=0.1)
    parser.add_argument("--displacement_scale", type=float, default=50.0)
    parser.add_argument("--voxel_size", type=float, default=0.3)
    args = parser.parse_args()
    main(args)
