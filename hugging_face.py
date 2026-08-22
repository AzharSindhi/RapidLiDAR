"""Push the trained RapidLiDAR (coarse) model to the Hugging Face Hub.

Uses `huggingface_hub`'s `PyTorchModelHubMixin` (see `RapidLiDARHubModel` in
`rapidlidar/models/hub.py`) so the model can be reloaded with a single
`from_pretrained` call. The data ranges are embedded into `config.json` so
`from_pretrained` works from any directory, without `configs/rapidlidar.yaml`
on disk.

Note: `PyTorchModelHubMixin` uploads only the weights + `config.json`; it does
NOT upload the model code. Users must have the `rapidlidar` package installed
(`pip install git+<repo>`), because the architecture also needs the compiled
mmcv deformable-attention op. See the generated model card for details.

Run:

    export CUDA_HOME=/path/to/cuda-12.4   # only needed to build/load the checkpoint
    python hugging_face.py
"""

import os
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi, login

from rapidlidar.models.hub import GITHUB_URL, TAGS, RapidLiDARHubModel

# --- configuration ----------------------------------------------------------
REPO_ID = "Azhar88/RapidLiDAR-coarse"
CHECKPOINT = "checkpoints/rapidlidar_vox_0.3_best.pth"
LOCAL_DIR = "RapidLiDAR-coarse"          # where save_pretrained writes
PRIVATE = False


def load_model():
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = dict(ckpt["hyper_parameters"])

    # Embed the data ranges so `from_pretrained` never needs the YAML file.
    with open(config["config_path"], "r") as f:
        data = yaml.safe_load(f)["data"]
    config.update(
        x_range=data["x_range"], y_range=data["y_range"],
        z_range=data["z_range"], num_points=data["num_points"],
    )

    model = RapidLiDARHubModel(**config)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    print(f"Loaded {CHECKPOINT}: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Config: {config}")
    return model, config


def model_card() -> str:
    tags_yaml = "\n".join(f"  - {t}" for t in TAGS)
    return f"""---
license: mit
library_name: pytorch
tags:
{tags_yaml}
---

# RapidLiDAR (coarse)

Single-pass LiDAR scene completion (coarse model, voxel size `0.3`), trained on
SemanticKITTI.

Code, installation, and full documentation:
[{GITHUB_URL}]({GITHUB_URL})

## Usage

```python
from rapidlidar.models.hub import RapidLiDARHubModel

model = RapidLiDARHubModel.from_pretrained("{REPO_ID}")
model.eval()

# x_partial: (B, N, 3) partial point cloud
output = model(x_partial, up_factor=10)
completed_points = output.points  # (B, N * up_factor, 3)
```
"""


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_file = Path("~/.cache/huggingface/token").expanduser()
        if token_file.exists():
            token = token_file.read_text().strip()
    login(token=token)

    model, _ = load_model()

    # Save locally, then push weights + config + card to the Hub.
    model.save_pretrained(LOCAL_DIR)
    model.push_to_hub(REPO_ID, private=PRIVATE)

    api = HfApi()
    # Minimal model card that points to the GitHub repo.
    api.upload_file(
        path_or_fileobj=model_card().encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add model card",
    )
    # Original Lightning checkpoint for use with the repo's tooling.
    api.upload_file(
        path_or_fileobj=CHECKPOINT,
        path_in_repo="rapidlidar_vox_0.3_best.pth",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add original Lightning checkpoint",
    )

    print(f"\nDone: https://huggingface.co/{REPO_ID}")

    # Sanity check: reload from the Hub.
    reloaded = RapidLiDARHubModel.from_pretrained(REPO_ID)
    reloaded.eval()
    print(f"Reloaded from hub OK: {sum(p.numel() for p in reloaded.parameters()):,} params")


if __name__ == "__main__":
    main()
