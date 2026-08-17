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

from rapidlidar.models.refinement import RefinementNetwork


def main(args):
    torch.set_float32_matmul_precision("medium")

    model = RefinementNetwork(
        base_model_checkpoint=args.base_model_checkpoint,
        config_path=args.config_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_stages=args.num_stages,
        kappa=args.kappa,
        attn_chunk_size=args.attn_chunk_size,
        use_grad_checkpoint=args.use_grad_checkpoint,
        fast_exp=args.fast_exp,
        fast_exp_factor=args.fast_exp_factor,
        completion_indices_file=args.completion_indices_file,
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

    if args.test:
        callbacks = []
    elif args.no_save_ckpt:
        callbacks = [lr_monitor]
    else:
        callbacks = [checkpoint_callback, lr_monitor]

    devices = -1 if args.device == -1 else [args.device]

    trainer = pl.Trainer(
        default_root_dir=run_ckpt_dir,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=devices,
        strategy="ddp_find_unused_parameters_false",
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        enable_checkpointing=not args.no_save_ckpt,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        overfit_batches=args.overfit_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
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
    parser.add_argument("--config_path", type=str, default="configs/rapidlidar_refine.yaml")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--experiment", "-e", type=str, default="rapidlidar_refine")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--overfit_batches", type=int, default=0)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=100)
    parser.add_argument("--limit_test_batches", type=int, default=None)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", "-d", type=int, default=-1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--num_stages", type=int, default=4)
    parser.add_argument("--kappa", type=int, default=6)
    parser.add_argument("--fast_exp", default=False, action="store_true")
    parser.add_argument("--fast_exp_factor", type=int, default=10)
    parser.add_argument("--fast_dev_run", default=False, action="store_true")
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-5)
    parser.add_argument("--attn_chunk_size", type=int, default=131072)
    parser.add_argument("--use_grad_checkpoint", default=True, action=argparse.BooleanOptionalAction)
    _sc = parser.add_mutually_exclusive_group()
    _sc.add_argument("--save_ckpt", dest="no_save_ckpt", action="store_false", default=False)
    _sc.add_argument("--no_save_ckpt", dest="no_save_ckpt", action="store_true")
    parser.add_argument("--completion_indices_file", type=str, default=None)
    parser.add_argument("--base_model_checkpoint", "-bc", type=str, required=True)
    args = parser.parse_args()
    main(args)
