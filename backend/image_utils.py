"""Pipeline Lite: Haar face → validate → crop → CLAHE-ish (Y eq) → letterbox nền xanh (không rembg)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
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


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _detect_faces_haar(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    dets = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    faces: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in dets:
        faces.append((int(x), int(y), int(x + w), int(y + h)))
    return faces


def _face_center_h_check(face_xyxy: Tuple[int, int, int, int], img_w: int, tolerance: float = 0.12) -> Tuple[bool, str]:
    x1, y1, x2, y2 = face_xyxy
    face_cx = (x1 + x2) / 2.0
    img_cx = img_w / 2.0
    delta = abs(face_cx - img_cx) / img_w
    if delta <= tolerance:
        return True, "Khuôn mặt nằm gần trung tâm theo chiều ngang."
    return False, "Khuôn mặt lệch khỏi trung tâm theo chiều ngang."


def _face_height_ratio(face_xyxy: Tuple[int, int, int, int], img_h: int) -> Tuple[bool, str, float]:
    x1, y1, x2, y2 = face_xyxy
    fh = max(1, y2 - y1)
    r = float(fh / max(1, img_h))
    if 0.50 <= r <= 0.70:
        return True, f"Khuôn mặt chiếm ≈{r*100:.1f}% chiều cao ảnh (đạt).", r
    return False, f"Khuôn mặt chiếm ≈{r*100:.1f}% chiều cao ảnh (chưa đạt).", r


def _brightness_contrast(bgr: np.ndarray) -> Tuple[float, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)), float(np.std(gray))


def _bc_check(brightness: float, contrast: float) -> Tuple[bool, List[str]]:
    warnings: List[str] = []
    ok = True
    if brightness < 80:
        ok = False
        warnings.append("Ảnh quá tối (độ sáng thấp).")
    elif brightness > 190:
        ok = False
        warnings.append("Ảnh quá sáng / bị cháy.")
    if contrast < 25:
        ok = False
        warnings.append("Ảnh bị mờ / tương phản thấp.")
    return ok, warnings


def _border_bg_check(bgr: np.ndarray, border_pct: float = 0.08) -> Tuple[bool, str]:
    h, w = bgr.shape[:2]
    bw = max(2, int(w * border_pct))
    bh = max(2, int(h * border_pct))
    border = np.concatenate(
        [
            bgr[0:bh, :, :].reshape(-1, 3),
            bgr[h - bh : h, :, :].reshape(-1, 3),
            bgr[:, 0:bw, :].reshape(-1, 3),
            bgr[:, w - bw : w, :].reshape(-1, 3),
        ],
        axis=0,
    )
    std = border.astype(np.float32).std(axis=0).mean()
    if std < 18.0:
        return True, f"Nền tương đối đơn sắc (độ lệch ~{std:.1f})."
    return False, f"Nền có thể không đơn sắc (độ lệch ~{std:.1f})."


def _compute_crop_rect(
    img_w: int,
    img_h: int,
    face_xyxy: Tuple[int, int, int, int],
    aspect: float,
    target_face_height_frac: float = 0.58,
    headroom_frac: float = 0.18,
) -> Tuple[int, int, int, int]:
    fx1, fy1, fx2, fy2 = face_xyxy
    face_h = max(1, fy2 - fy1)
    face_cx = (fx1 + fx2) / 2.0
    face_cy = (fy1 + fy2) / 2.0
    crop_h = int(face_h / _clamp(target_face_height_frac, 0.45, 0.75))
    crop_w = int(crop_h * aspect)
    desired_top = int(face_cy - (0.5 - headroom_frac) * crop_h)
    desired_left = int(face_cx - crop_w / 2)
    x1, y1 = desired_left, desired_top
    x2, y2 = x1 + crop_w, y1 + crop_h
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        dx = x2 - img_w
        x1 -= dx
        x2 = img_w
    if y2 > img_h:
        dy = y2 - img_h
        y1 -= dy
        y2 = img_h
    x1 = int(_clamp(x1, 0, img_w - 1))
    y1 = int(_clamp(y1, 0, img_h - 1))
    x2 = int(_clamp(x2, x1 + 1, img_w))
    y2 = int(_clamp(y2, y1 + 1, img_h))
    return x1, y1, x2, y2


def _equalize_y(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)


def _resize_to_standard(bgr: np.ndarray, ratio_name: str) -> np.ndarray:
    if ratio_name == "3x4":
        out_w, out_h = 600, 800
    else:
        out_w, out_h = 600, 900
    return cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def _letterbox_on_blue(
    bgr: np.ndarray,
    blue_rgb: Tuple[int, int, int],
    out_w: int,
    out_h: int,
    margin: float = 0.06,
) -> np.ndarray:
    """Nền xanh full khung; ảnh scale vào vùng giữa (còn lề xanh)."""
    h, w = bgr.shape[:2]
    inner_w = int(out_w * (1 - 2 * margin))
    inner_h = int(out_h * (1 - 2 * margin))
    inner_w = max(1, inner_w)
    inner_h = max(1, inner_h)
    scale = min(inner_w / w, inner_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:, :] = (blue_rgb[2], blue_rgb[1], blue_rgb[0])
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def process_portrait_lite(
    pil_img: Image.Image,
    ratio: str = "3x4",
    blue_rgb: Tuple[int, int, int] = (0, 91, 196),
) -> ProcessResult:
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, CheckResult] = {}

    bgr0 = _pil_to_bgr(pil_img)
    h0, w0 = bgr0.shape[:2]

    faces = _detect_faces_haar(bgr0)
    if len(faces) == 0:
        errors.append("Không tìm thấy khuôn mặt.")
        checks["Khuôn mặt"] = CheckResult(False, "Không phát hiện được khuôn mặt (Haar).")
        return ProcessResult(status="FAILED", errors=errors, warnings=warnings, checks=checks, processed_image=None)
    if len(faces) > 1:
        errors.append("Có nhiều hơn 1 khuôn mặt trong ảnh.")
        checks["Khuôn mặt"] = CheckResult(False, f"Phát hiện {len(faces)} khuôn mặt.")
        return ProcessResult(status="FAILED", errors=errors, warnings=warnings, checks=checks, processed_image=None)

    fx1, fy1, fx2, fy2 = faces[0]
    checks["Khuôn mặt"] = CheckResult(True, "Phát hiện đúng 1 khuôn mặt (Haar).")

    ok_c, msg_c = _face_center_h_check((fx1, fy1, fx2, fy2), w0)
    checks["Vị trí (giữa khung)"] = CheckResult(ok_c, msg_c)
    if not ok_c:
        warnings.append(msg_c)

    ok_r, msg_r, _ = _face_height_ratio((fx1, fy1, fx2, fy2), h0)
    checks["Tỷ lệ khuôn mặt"] = CheckResult(ok_r, msg_r)
    if not ok_r:
        warnings.append(msg_r)

    br, ct = _brightness_contrast(bgr0)
    ok_bc, bc_warns = _bc_check(br, ct)
    checks["Ánh sáng & Tương phản"] = CheckResult(ok_bc, f"Độ sáng ~{br:.0f}, tương phản ~{ct:.0f}.")
    warnings.extend(bc_warns)

    ok_bg, msg_bg = _border_bg_check(bgr0)
    checks["Nền đơn sắc"] = CheckResult(ok_bg, msg_bg)
    if not ok_bg:
        warnings.append(msg_bg)

    aspect = 3 / 4 if ratio == "3x4" else 2 / 3
    x1, y1, x2, y2 = _compute_crop_rect(w0, h0, (fx1, fy1, fx2, fy2), aspect=aspect)
    cropped = bgr0[y1:y2, x1:x2].copy()
    if cropped.size == 0:
        errors.append("Không crop được vùng ảnh.")
        return ProcessResult(status="FAILED", errors=errors, warnings=warnings, checks=checks, processed_image=None)

    cropped_eq = _equalize_y(cropped)
    out_bgr = _resize_to_standard(cropped_eq, ratio_name=ratio)
    out_w, out_h = out_bgr.shape[1], out_bgr.shape[0]
    final_bgr = _letterbox_on_blue(out_bgr, blue_rgb=blue_rgb, out_w=out_w, out_h=out_h)

    checks["Nền xanh (Lite)"] = CheckResult(
        True,
        "Ghép ảnh lên nền xanh (letterbox, không dùng AI tách nền).",
    )

    pil_out = _bgr_to_pil(final_bgr).convert("RGB")
    return ProcessResult(status="OK", errors=errors, warnings=warnings, checks=checks, processed_image=pil_out)


def pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(_clamp(quality, 60, 100)), optimize=True)
    return buf.getvalue()
