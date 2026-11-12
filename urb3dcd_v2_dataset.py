"""
Dedicated Urb3DCD-V2 dataset entry points.

The actual implementation lives in cached_dataset.py so the SLPCCD and
Urb3DCD-V2 training scripts share one tested sampler implementation.
"""

from cached_dataset import (
    CDDatasetUrb3DCDV2ForPipeline,
    sample_label1_multiclass_cylinder_pair,
    sample_label1_multiclass_pair,
)


URB3DCD_V2_CLASS_LABELS = [
    "unchanged",
    "new_building",
    "demolition",
    "new_vegetation",
    "vegetation_growth",
    "missing_vegetation",
    "mobile_objects",
]


class Urb3DCDV2Dataset(CDDatasetUrb3DCDV2ForPipeline):
    """Paper-facing alias for the Urb3DCD-V2 all-label dataset reader."""


__all__ = [
    "Urb3DCDV2Dataset",
    "URB3DCD_V2_CLASS_LABELS",
    "sample_label1_multiclass_cylinder_pair",
    "sample_label1_multiclass_pair",
]
