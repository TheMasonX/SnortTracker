"""
Dataset collection, slicing, labeling, and preprocessing for SnortTracker.

Modules
-------
- ``manifest`` — CSV manifest for tracking recordings and labels
- ``slicer`` — Audio window slicing per the audio contract
- ``labeler`` — Label policy enforcement and session splitting
- ``preprocessor`` — Batch feature extraction for training
"""

from dataset.manifest import Manifest, ManifestEntry, LabelStatus, DataSplit
from dataset.slicer import Slicer, SlicedWindow
from dataset.labeler import LabelPolicy, SessionSplitter, validate_manifest_labels
from dataset.preprocessor import Preprocessor

__all__ = [
    "Manifest",
    "ManifestEntry",
    "LabelStatus",
    "DataSplit",
    "Slicer",
    "SlicedWindow",
    "LabelPolicy",
    "SessionSplitter",
    "validate_manifest_labels",
    "Preprocessor",
]
