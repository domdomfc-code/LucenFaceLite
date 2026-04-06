"""Lite: không crop, không nền xanh — chỉ chuẩn hóa RGB + xuất JPEG."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image


@dataclass
class CheckResult:
    ok: bool
    message: str


@dataclass
class ProcessResult:
    status: str
    errors: List[str]
    warnings: List[str]
    checks: Dict[str, CheckResult]
    processed_image: Optional[Image.Image]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def prepare_lite_output(pil_img: Image.Image) -> ProcessResult:
    """Giữ nguyên khung ảnh; không cắt, không chèn nền."""
    img = pil_img.convert("RGB")
    checks = {
        "Chế độ Lite": CheckResult(
            True,
            "Ảnh giữ nguyên — không tự động cắt, không chèn nền xanh.",
        ),
    }
    return ProcessResult(
        status="OK",
        errors=[],
        warnings=[],
        checks=checks,
        processed_image=img,
    )


def pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(_clamp(quality, 60, 100)), optimize=True)
    return buf.getvalue()
