# LucenFace Lite (p2clite)

Deploy **rất nhẹ** trên Streamlit Cloud: chỉ `streamlit` + `pillow`.

- **Không** tự động cắt ảnh, **không** chèn nền xanh — chỉ xem và xuất JPG/ZIP (ảnh giữ nguyên khung).
- **Không** OpenCV / MediaPipe / rembg.

## Chạy local

```bash
cd p2clite
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

- **Main file** `app.py`
- **Requirements** `requirements.txt`
- **Không** cần `packages.txt`.
