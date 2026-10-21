# Import packages
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from segmentation_inference.model import binary_hake_model
from segmentation_inference.utils import get_MVBS_tensor


def test_binary_hake_model_inference():
    """
    Test MVBS tensor conversion and binary hake model inference.
    """
    # Setup paths
    base_path = Path("segmentation_inference/test_data")
    MVBS_ds_path = base_path / "MVBS_ds_test.zarr"
    binary_hake_model_path = base_path / "binary_hake_model_weight_test.ckpt"
    softmax_score_ds_path = base_path / "softmax_score_ds_test.zarr"

    # Read MVBS dataset and convert to tensor
    MVBS_ds = xr.open_zarr(MVBS_ds_path)
    max_depth = 590
    MVBS_tensor = get_MVBS_tensor(MVBS_ds, max_depth=max_depth)

    # Load model using test weight
    model = binary_hake_model().eval()
    model.load_state_dict(
        torch.load(binary_hake_model_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]
    )

    # Pass MVBS tensor into model
    output_dict = model.forward(MVBS_tensor, softmax_temperature=1.0)

    # Check output shapes
    assert output_dict["backbone_output"].shape == (
        2048,
        74,
        63,
    )  # (channel, pre-interp depth, pre-interp ping time)
    assert output_dict["classifier_output"].shape == (
        2,
        74,
        63,
    )  # (class, pre-interp depth, pre-interp ping time)
    assert output_dict["interpolated_output"].shape == (2, 591, 500)  # (class, depth, ping time)
    assert output_dict["softmax_output"].shape == (2, 591, 500)  # (class, depth, ping time)

    # Check softmax output
    softmax_score_da = xr.DataArray(
        output_dict["softmax_output"].detach(),
        coords={
            "scatterer_class": ["background", "hake"],
            "depth": MVBS_ds["depth"].sel(depth=slice(None, max_depth)).values,
            "ping_time": MVBS_ds["ping_time"].values,
        },
        name="softmax_score",
    )
    softmax_score_da_test = xr.open_zarr(softmax_score_ds_path)["softmax_score"]
    assert softmax_score_da["depth"].equals(softmax_score_da_test["depth"])
    assert softmax_score_da["ping_time"].equals(softmax_score_da_test["ping_time"])
    assert softmax_score_da["scatterer_class"].equals(softmax_score_da_test["scatterer_class"])
    assert np.allclose(
        softmax_score_da.values,
        softmax_score_da_test.values,
        atol=0.01,
    )
