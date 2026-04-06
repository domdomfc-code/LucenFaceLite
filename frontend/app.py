from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.image_utils import ProcessResult, pil_to_jpeg_bytes, process_portrait_lite

APP_TITLE = "LucenFace Lite"
APP_BUILD = "1.1-dnn-ssd"
BLUE = "#005BC4"
BG = "#F6F9FF"


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
          :root {{
            --blue: {BLUE};
            --bg: {BG};
            --text: #0f172a;
            --muted: #64748b;
            --border: rgba(2, 6, 23, 0.10);
            --shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
          }}
          .stApp {{
            background:
              radial-gradient(1000px 500px at 20% 0%, rgba(0, 91, 196, 0.14), rgba(0,0,0,0) 60%),
              var(--bg);
          }}
          section.main > div {{ padding-top: 1.1rem; }}
          header, footer {{ visibility: hidden; height: 0; }}
          .topbar {{
            display: flex; align-items: center; justify-content: space-between; gap: 12px;
            background: rgba(255,255,255,0.75); border: 1px solid var(--border);
            border-radius: 16px; padding: 12px 14px; box-shadow: var(--shadow);
          }}
          .brand {{ display: flex; align-items: center; gap: 10px; font-weight: 900; color: var(--text); }}
          .brand-badge {{
            width: 34px; height: 34px; border-radius: 10px;
            background: linear-gradient(135deg, var(--blue), #38bdf8);
          }}
          .pill {{
            padding: 7px 10px; border: 1px solid var(--border); border-radius: 999px;
            background: rgba(255,255,255,0.9); font-weight: 700; color: var(--muted); font-size: 0.85rem;
          }}
          .app-title {{ font-size: 1.75rem; font-weight: 900; color: var(--text); margin: 0.3rem 0; }}
          .app-subtitle {{ color: var(--muted); font-weight: 600; margin-bottom: 0.8rem; }}
          .card {{
            background: #fff; border: 1px solid var(--border); border-radius: 16px;
            padding: 14px; box-shadow: var(--shadow);
          }}
          .badge-ok {{ display:inline-block; padding:2px 10px; border-radius:999px;
            background: rgba(22,163,74,0.12); color:#166534; font-weight:700; font-size:0.85rem; }}
          .badge-fail {{ display:inline-block; padding:2px 10px; border-radius:999px;
            background: rgba(220,38,38,0.12); color:#991b1b; font-weight:700; font-size:0.85rem; }}
          .checklist {{ margin-top: 6px; }}
          .check-item {{ display:flex; gap:8px; margin:4px 0; align-items:baseline; }}
          .check-name {{ font-weight:700; min-width: 160px; }}
          .muted {{ color: var(--muted); }}
          [data-testid="stFileUploader"] > div {{
            border: 2px dashed rgba(0, 91, 196, 0.35);
            background: rgba(255,255,255,0.85);
            border-radius: 18px; padding: 16px;
          }}
          .stButton > button, .stDownloadButton > button {{ border-radius: 12px; font-weight: 800; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _checklist_html(checks: Dict[str, Dict[str, str]]) -> str:
    parts = ['<div class="checklist">']
    for name, payload in checks.items():
        ok = payload["ok"]
        icon = "✅" if ok else "❌"
        parts.append(
            f'<div class="check-item"><div>{icon}</div><div class="check-name">{name}</div><div>{payload["message"]}</div></div>'
        )
    parts.append("</div>")
    return "\n".join(parts)


def _result_to_checks_dict(res: ProcessResult) -> Dict[str, Dict[str, str]]:
    return {k: {"ok": bool(v.ok), "message": str(v.message)} for k, v in res.checks.items()}


def _make_zip(files: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            z.writestr(name, data)
    return buf.getvalue()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="⚡", layout="wide")
    _inject_css()

    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">
            <div class="brand-badge"></div>
            <div>
              <div style="font-size:1.05rem;font-weight:900;">{APP_TITLE}</div>
              <div class="muted" style="font-size:0.8rem;font-weight:700;">Build {APP_BUILD} — deploy nhanh</div>
            </div>
          </div>
          <div><span class="pill">OpenCV DNN / Haar</span><span class="pill">Không rembg</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="app-title">Chuẩn hóa ảnh chân dung (Lite)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Bản nhẹ: ưu tiên phát hiện mặt OpenCV DNN (SSD), fallback Haar; CLAHE; không rembg — viền nền xanh tùy chọn.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Cài đặt")
        ratio = st.selectbox("Tỷ lệ đầu ra", ["3x4", "4x6"], index=0)
        st.caption("Tối đa 50 ảnh/lần.")
        st.markdown("---")
        st.markdown("#### Độ chính xác (DNN)")
        dnn_conf = st.slider(
            "Ngưỡng tin cậy SSD",
            min_value=0.30,
            max_value=0.65,
            value=0.45,
            step=0.05,
            help="Cao hơn → ít box nhầm (dễ bỏ sót mặt nhỏ / nghiêng). Thấp hơn → nhạy hơn.",
        )
        lb_margin = st.slider(
            "Lề nền xanh (letterbox)",
            min_value=0.0,
            max_value=0.10,
            value=0.0,
            step=0.01,
            format="%.2f",
            help="0 = ảnh fill đủ khung chuẩn (không viền). >0 = thu ảnh lại để lộ nền xanh.",
        )
        blue_hex = st.color_picker("Màu nền (khi có lề)", value=BLUE)

    blue_rgb = tuple(int(blue_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    st.markdown("### Kéo & thả ảnh")
    uploads = st.file_uploader(
        "Kéo và thả tệp vào đây hoặc bấm để chọn",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploads:
        st.info("Upload ít nhất 1 ảnh.")
        return
    if len(uploads) > 50:
        st.error("Tối đa 50 ảnh.")
        return

    if not st.button("Bắt đầu xử lý", type="primary"):
        st.caption("Bấm **Bắt đầu xử lý** để chạy.")
        return

    progress = st.progress(0)
    zip_items: List[Tuple[str, bytes]] = []

    for idx, up in enumerate(uploads, start=1):
        progress.progress(min(100, int((idx - 1) / max(len(uploads), 1) * 100)))
        raw = up.read()
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            st.error(f"Không đọc được: `{up.name}`")
            continue

        res = process_portrait_lite(
            pil,
            ratio=ratio,
            blue_rgb=blue_rgb,
            dnn_min_confidence=float(dnn_conf),
            letterbox_margin=float(lb_margin),
        )

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.05, 1.05, 1.25], gap="large")
            with c1:
                st.markdown("**Original**")
                st.image(pil, caption=up.name, use_container_width=True)
            with c2:
                st.markdown("**Trạng thái**")
                if res.status == "OK":
                    st.markdown('<span class="badge-ok">OK</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-fail">FAILED</span>', unsafe_allow_html=True)
                if res.errors:
                    st.error("\n".join(res.errors))
                if res.warnings:
                    st.warning("\n".join(res.warnings))
                st.markdown("**Checklist**")
                st.markdown(_checklist_html(_result_to_checks_dict(res)), unsafe_allow_html=True)
            with c3:
                st.markdown("**Processed**")
                if res.processed_image is None:
                    st.info("Không xử lý được.")
                else:
                    st.image(res.processed_image, use_container_width=True)
                    out_bytes = pil_to_jpeg_bytes(res.processed_image, quality=95)
                    base = up.name.rsplit(".", 1)[0] if "." in up.name else up.name
                    zip_name = f"{idx:03d}_{base}_lite.jpg"
                    zip_items.append((zip_name, out_bytes))
                    st.download_button(
                        "Download JPG",
                        data=out_bytes,
                        file_name=f"{base}_lite.jpg",
                        mime="image/jpeg",
                        key=f"dl_{idx}",
                    )
            st.markdown("</div>", unsafe_allow_html=True)

    progress.progress(100)
    if zip_items:
        st.download_button(
            label=f"Download ZIP ({len(zip_items)} ảnh)",
            data=_make_zip(zip_items),
            file_name="lucenface_lite.zip",
            mime="application/zip",
            type="primary",
            key="dl_zip",
        )


if __name__ == "__main__":
    main()
