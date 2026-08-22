"""Hugging Face Hub wrapper for RapidLiDAR.

`RapidLiDARHubModel` is a thin subclass that adds `save_pretrained` /
`from_pretrained` (via `PyTorchModelHubMixin`) without touching the core model.

    from rapidlidar.models.hub import RapidLiDARHubModel

    model = RapidLiDARHubModel.from_pretrained("Azhar88/RapidLiDAR-coarse")
    model.eval()
    output = model(x_partial, up_factor=10)
"""

from huggingface_hub import PyTorchModelHubMixin

from rapidlidar.models.rapidlidar import RapidLiDAR

GITHUB_URL = "https://github.com/AzharSindhi/RapidLiDAR"
TAGS = [
    "lidar",
    "point-cloud",
    "point-cloud-completion",
    "scene-completion",
    "3d",
    "semantickitti",
    "autonomous-driving",
]


class RapidLiDARHubModel(
    RapidLiDAR,
    PyTorchModelHubMixin,
    repo_url=GITHUB_URL,
    license="mit",
    tags=TAGS,
    library_name="pytorch",
):
    """`RapidLiDAR` augmented with `save_pretrained` / `from_pretrained`."""
