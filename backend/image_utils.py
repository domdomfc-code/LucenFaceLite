"""Pipeline Lite: DNN (SSD) hoặc Haar → validate → crop → CLAHE (Y) → nền xanh tùy letterbox (không rembg)."""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_PROTO = _MODEL_DIR / "deploy.prototxt"
_WEIGHTS = _MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
_WEIGHTS_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)


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


def _ensure_dnn_model_files() -> bool:
    """Tải prototxt + caffemodel nếu thiếu. Trả False nếu không đọc được (fallback Haar)."""
    try:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if not _PROTO.exists() or _PROTO.stat().st_size < 1000:
            urllib.request.urlretrieve(_PROTO_URL, _PROTO)  # noqa: S310
        if not _WEIGHTS.exists() or _WEIGHTS.stat().st_size < 1_000_000:
            urllib.request.urlretrieve(_WEIGHTS_URL, _WEIGHTS)  # noqa: S310
        return _PROTO.exists() and _WEIGHTS.exists() and _WEIGHTS.stat().st_size > 1_000_000
    except Exception:
        return False


_dnn_net: Optional[cv2.dnn_Net] = None


def _get_dnn_face_net() -> Optional[cv2.dnn_Net]:
    global _dnn_net
    if _dnn_net is not None:
        return _dnn_net
    if not _ensure_dnn_model_files():
        return None
    try:
        _dnn_net = cv2.dnn.readNetFromCaffe(str(_PROTO), str(_WEIGHTS))
        return _dnn_net
    except Exception:
        return None


def _xyxy_to_xywh(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _detect_faces_dnn(
    bgr: np.ndarray,
    min_confidence: float,
) -> List[Tuple[int, int, int, int, float]]:
    net = _get_dnn_face_net()
    if net is None:
        return []
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 117.0, 123.0),
    )
    net.setInput(blob)
    det = net.forward()
    boxes_xywh: List[Tuple[int, int, int, int]] = []
    scores: List[float] = []
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < min_confidence:
            continue
        bx = det[0, 0, i, 3:7] * np.array([w, h, w, h], dtype=np.float32)
        x1, y1, x2, y2 = [int(round(v)) for v in bx]
        x1 = int(_clamp(x1, 0, w - 1))
        y1 = int(_clamp(y1, 0, h - 1))
        x2 = int(_clamp(x2, x1 + 1, w))
        y2 = int(_clamp(y2, y1 + 1, h))
        ww, hh = _xyxy_to_xywh(x1, y1, x2, y2)[2:]
        boxes_xywh.append((x1, y1, ww, hh))
        scores.append(conf)

    if not boxes_xywh:
        return []

    idx = cv2.dnn.NMSBoxes(boxes_xywh, scores, min_confidence, 0.4)
    if idx is None or (hasattr(idx, "__len__") and len(idx) == 0):
        return []
    flat = idx.flatten() if hasattr(idx, "flatten") else np.array(idx).flatten()
    out: List[Tuple[int, int, int, int, float]] = []
    for j in flat:
        x1, y1, ww, hh = boxes_xywh[int(j)]
        sc = scores[int(j)]
        out.append((x1, y1, x1 + ww, y1 + hh, sc))
    out.sort(key=lambda t: (t[4] * (t[2] - t[0]) * (t[3] - t[1]), t[4]), reverse=True)
    return out


def _detect_faces_haar(bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return []

    h, w = gray.shape[:2]
    min_face = max(36, int(min(h, w) * 0.06))

    def run(g: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return list(
            face_cascade.detectMultiScale(
                g,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(min_face, min_face),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        )

    rects = run(gray)
    if not rects and max(h, w) > 960:
        scale = 960 / max(h, w)
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        rects = run(small)
        out: List[Tuple[int, int, int, int, float]] = []
        for (x, y, ww, hh) in rects:
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + ww) / scale)
            y2 = int((y + hh) / scale)
            out.append((x1, y1, x2, y2, 0.45))
        return _nms_xyxy_scores(out, 0.3)

    out = []
    for (x, y, ww, hh) in rects:
        out.append((int(x), int(y), int(x + ww), int(y + hh), 0.45))
    return _nms_xyxy_scores(out, 0.3)


def _nms_xyxy_scores(
    faces: List[Tuple[int, int, int, int, float]],
    iou_thresh: float,
) -> List[Tuple[int, int, int, int, float]]:
    if len(faces) <= 1:
        return faces
    if len(faces) > 30:
        faces = sorted(faces, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]), reverse=True)[:30]
    boxes = [[f[0], f[1], f[2] - f[0], f[3] - f[1]] for f in faces]
    scores = [f[4] for f in faces]
    idx = cv2.dnn.NMSBoxes(boxes, scores, 0.1, iou_thresh)
    if idx is None or len(idx) == 0:
        return faces
    flat = idx.flatten() if hasattr(idx, "flatten") else np.array(idx).flatten()
    return [faces[int(j)] for j in flat]


def _detect_faces(
    bgr: np.ndarray,
    dnn_min_conf: float,
) -> Tuple[List[Tuple[int, int, int, int]], str]:
    """Trả về danh sách bbox (không score) và nhãn nguồn."""
    dets = _detect_faces_dnn(bgr, min_confidence=dnn_min_conf)
    if dets:
        faces = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in dets]
        return faces, "OpenCV DNN (SSD)"
    dets2 = _detect_faces_haar(bgr)
    if dets2:
        faces = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in dets2]
        return faces, "Haar + CLAHE"
    return [], "—"


def _expand_face_for_crop(
    face_xyxy: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad_side: float = 0.14,
    pad_top: float = 0.22,
    pad_bottom: float = 0.10,
) -> Tuple[int, int, int, int]:
    """Mở bbox: Haar/DNN thường bó sát; thêm trán / má."""
    x1, y1, x2, y2 = face_xyxy
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)
    nx1 = int(x1 - pad_side * fw)
    ny1 = int(y1 - pad_top * fh)
    nx2 = int(x2 + pad_side * fw)
    ny2 = int(y2 + pad_bottom * fh)
    nx1 = int(_clamp(nx1, 0, img_w - 1))
    ny1 = int(_clamp(ny1, 0, img_h - 1))
    nx2 = int(_clamp(nx2, nx1 + 1, img_w))
    ny2 = int(_clamp(ny2, ny1 + 1, img_h))
    return nx1, ny1, nx2, ny2


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


def _equalize_y_clahe(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y2 = clahe.apply(y)
    return cv2.cvtColor(cv2.merge([y2, cr, cb]), cv2.COLOR_YCrCb2BGR)


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
    margin: float,
) -> np.ndarray:
    if margin <= 0:
        return cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
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
    dnn_min_confidence: float = 0.45,
    letterbox_margin: float = 0.0,
) -> ProcessResult:
    """
    dnn_min_confidence: ngưỡng SSD (0.35–0.55). Cao hơn → ít báo nhầm nhiều mặt.
    letterbox_margin: 0 = fill đủ khung 600×800 (không lề xanh). >0 = viền nền xanh.
    """
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, CheckResult] = {}

    bgr0 = _pil_to_bgr(pil_img)
    h0, w0 = bgr0.shape[:2]

    faces, det_label = _detect_faces(bgr0, dnn_min_conf=dnn_min_confidence)
    if len(faces) == 0:
        errors.append("Không tìm thấy khuôn mặt.")
        checks["Khuôn mặt"] = CheckResult(False, "Không phát hiện được khuôn mặt.")
        return ProcessResult(status="FAILED", errors=errors, warnings=warnings, checks=checks, processed_image=None)
    if len(faces) > 1:
        errors.append("Có nhiều hơn 1 khuôn mặt trong ảnh.")
        checks["Khuôn mặt"] = CheckResult(False, f"Phát hiện {len(faces)} khuôn mặt.")
        return ProcessResult(status="FAILED", errors=errors, warnings=warnings, checks=checks, processed_image=None)

    fx1, fy1, fx2, fy2 = faces[0]
    checks["Khuôn mặt"] = CheckResult(True, f"Đúng 1 mặt ({det_label}).")

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

    ex1, ey1, ex2, ey2 = _expand_face_for_crop((fx1, fy1, fx2, fy2), w0, h0)

    aspect = 3 / 4 if ratio == "3x4" else 2 / 3
    x1, y1, x2, y2 = _compute_crop_rect(w0, h0, (ex1, ey1, ex2, ey2), aspect=aspect)
    cropped = bgr0[y1:y2, x1:x2].copy()
    if cropped.size == 0:
        errors.append("Không crop được vùng ảnh.")
        return ProcessResult(status="FAILED", errors=errors, warnings=warnings, checks=checks, processed_image=None)

    cropped_eq = _equalize_y_clahe(cropped)
    out_bgr = _resize_to_standard(cropped_eq, ratio_name=ratio)
    out_w, out_h = out_bgr.shape[1], out_bgr.shape[0]
    final_bgr = _letterbox_on_blue(out_bgr, blue_rgb=blue_rgb, out_w=out_w, out_h=out_h, margin=letterbox_margin)

    if letterbox_margin > 0:
        msg_bg_lite = "Ghép letterbox lên nền xanh (không AI tách nền)."
    else:
        msg_bg_lite = "Fill khung chuẩn (không viền xanh — ảnh full 600×...)."
    checks["Nền xanh (Lite)"] = CheckResult(True, msg_bg_lite)

    pil_out = _bgr_to_pil(final_bgr).convert("RGB")
    return ProcessResult(status="OK", errors=errors, warnings=warnings, checks=checks, processed_image=pil_out)


def pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(_clamp(quality, 60, 100)), optimize=True)
    return buf.getvalue()
