from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.image_utils import ProcessResult, pil_to_jpeg_bytes, prepare_lite_output

APP_TITLE = "LucenFace Lite"
APP_BUILD = "2.0-no-crop-preview"
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
          header[data-testid="stHeader"] {{
            background: rgba(255, 255, 255, 0.55) !important;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
          }}
          .stDeployButton {{ display: none !important; }}
          footer {{ visibility: hidden; height: 0; }}
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


def _sidebar_reopen_button() -> None:
    components.html(
        """
        <div style="position:fixed;top:52px;left:8px;z-index:999999;">
        <button type="button" title="Mở cài đặt (sidebar)"
          onclick="(() => {
            try {
              const d = window.parent.document;
              const q = (s) => d.querySelector(s);
              (q('[data-testid="collapsedControl"]')
                || q('[data-testid="stSidebarCollapsedControl"]')
                || q('button[data-testid="baseButton-header"]')
                || q('header button[kind="header"]'))?.click();
            } catch (e) {}
          })()"
          style="font-size:1.05rem;line-height:1;padding:0.45rem 0.55rem;border-radius:10px;
                 border:1px solid rgba(15,23,42,0.12);background:rgba(255,255,255,0.96);
                 cursor:pointer;box-shadow:0 4px 14px rgba(15,23,42,0.12);color:#0f172a;">
          ☰
        </button>
        </div>
        """,
        height=52,
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
    _sidebar_reopen_button()

    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">
            <div class="brand-badge"></div>
            <div>
              <div style="font-size:1.05rem;font-weight:900;">{APP_TITLE}</div>
              <div class="muted" style="font-size:0.8rem;font-weight:700;">Build {APP_BUILD} — xem & tải nhanh</div>
            </div>
          </div>
          <div><span class="pill">Không OpenCV</span><span class="pill">Chỉ Pillow</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="app-title">Xem & xuất ảnh (Lite)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Không tự động cắt, không chèn nền xanh — ảnh giữ nguyên, chỉ xuất JPG/ZIP.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Cài đặt")
        jpeg_q = st.slider("Chất lượng JPEG xuất", min_value=60, max_value=100, value=95, step=5)
        st.caption("Tối đa 50 ảnh/lần.")

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

    zip_items: List[Tuple[str, bytes]] = []

    for idx, up in enumerate(uploads, start=1):
        raw = up.read()
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            st.error(f"Không đọc được: `{up.name}`")
            continue

        res = prepare_lite_output(pil)

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2 = st.columns([1.2, 1], gap="large")
            with c1:
                st.markdown("**Ảnh**")
                st.image(pil, caption=up.name, use_container_width=True)
            with c2:
                st.markdown("**Trạng thái**")
                st.markdown('<span class="badge-ok">OK</span>', unsafe_allow_html=True)
                st.markdown("**Thông tin**")
                st.markdown(_checklist_html(_result_to_checks_dict(res)), unsafe_allow_html=True)
                if res.processed_image is not None:
                    out_bytes = pil_to_jpeg_bytes(res.processed_image, quality=int(jpeg_q))
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
