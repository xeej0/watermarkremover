"""wmremover: AI-powered visible watermark detection and removal."""

from .detect import detect_watermark, estimate_batch_watermark
from .inpaint import ClassicalInpainter, LamaInpainter, get_inpainter
from .pipeline import remove_watermark, remove_watermark_batch

__all__ = [
    "detect_watermark",
    "estimate_batch_watermark",
    "ClassicalInpainter",
    "LamaInpainter",
    "get_inpainter",
    "remove_watermark",
    "remove_watermark_batch",
]
__version__ = "0.1.0"
