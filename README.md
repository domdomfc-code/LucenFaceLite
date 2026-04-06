# LucenFace Lite (p2clite)

Deploy **nhanh** trên Streamlit Cloud: chỉ `streamlit` + `opencv-python-headless` + `numpy` + `pillow`.

- Phát hiện mặt: **OpenCV Haar** (không MediaPipe).
- Không **rembg / onnxruntime** → cài đặt nhẹ, boot nhanh.
- Nền xanh: **letterbox** (không AI tách nền).

## Chạy local

```bash
cd p2clite
pip install -r requirements.txt
streamlit run app.py
```

## Tạo repository GitHub mới và push

1. Trên GitHub: **New repository** → tên ví dụ `p2clite` (public), **không** tick README.
2. Máy bạn:

```bash
cd E:\Code\p2clite
git init
git branch -M main
git add .
git commit -m "Initial commit: LucenFace Lite"
git remote add origin https://github.com/<USER>/p2clite.git
git push -u origin main
```

## Streamlit Cloud

- **Main file** `app.py`
- **Requirements** `requirements.txt` (mặc định)
- **Không** cần `packages.txt` (để apt nhẹ).
