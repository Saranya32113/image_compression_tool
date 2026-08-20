import streamlit as st
import cv2
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="HD Image Compression",
    page_icon="🖼️",
    layout="wide"
)

# =====================================================
# SIDEBAR SETTINGS
# =====================================================
with st.sidebar:
    st.markdown("## ⚙️ Compression Settings")

    compression_mode = st.radio(
        "Compression Mode",
        [
            "HD Clear (High Quality)",
            "Balanced (Recommended)",
            "Ultra (Maximum Compression)"
        ]
    )

    st.markdown("---")
    st.caption(
        "✔ Same Resolution\n"
        "✔ Smaller File Size\n"
        "✔ No Blur\n"
        "✔ Multiple Image Support"
    )

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<h1 style="text-align:center;">🖼️ HD Image Compression</h1>
<p style="text-align:center;color:gray;">
Google-style compression • Same resolution • Smaller size
</p>
<hr>
""", unsafe_allow_html=True)

# =====================================================
# MULTIPLE FILE UPLOAD
# =====================================================
uploaded_files = st.file_uploader(
    "📤 Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    # =================================================
    # QUALITY SETTINGS
    # =================================================
    if compression_mode.startswith("HD Clear"):
        start_quality = 90
    elif compression_mode.startswith("Balanced"):
        start_quality = 70
    else:
        start_quality = 50

    st.markdown("## Compression Results")

    # =================================================
    # PROCESS EACH IMAGE
    # =================================================
    for uploaded_file in uploaded_files:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Fix blue color issue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        orig_h, orig_w, _ = image_rgb.shape
        original_size_kb = uploaded_file.size / 1024

        # =================================================
        # SMART COMPRESSION
        # =================================================
        quality = start_quality
        compressed_bytes = None

        while quality > 20:

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, encoded_img = cv2.imencode(".jpg", image_bgr, encode_param)

            compressed_bytes = encoded_img.tobytes()
            final_size_kb = len(compressed_bytes) / 1024

            if final_size_kb < original_size_kb:
                break

            quality -= 5

        reduction = ((original_size_kb - final_size_kb) / original_size_kb) * 100
        safe_reduction = max(0, min(int(reduction), 100))

        # Decode compressed image
        compressed_np = np.frombuffer(compressed_bytes, np.uint8)
        compressed_bgr = cv2.imdecode(compressed_np, cv2.IMREAD_COLOR)
        compressed_rgb = cv2.cvtColor(compressed_bgr, cv2.COLOR_BGR2RGB)

        # =================================================
        # DISPLAY IMAGES
        # =================================================
        st.markdown(f"### 🖼️ {uploaded_file.name}")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image_rgb, use_container_width=True)
            st.write(f"📦 Original: **{original_size_kb:.2f} KB**")
            st.write(f"📐 {orig_w} × {orig_h}")

        with col2:
            st.image(compressed_rgb, use_container_width=True)
            st.write(f"📦 Compressed: **{final_size_kb:.2f} KB**")
            st.write(f"📐 {orig_w} × {orig_h}")

        # =================================================
        # COMPRESSION RESULT
        # =================================================
        st.progress(safe_reduction)

        if reduction > 0:
            st.success(f"🔥 Size reduced by **{reduction:.1f}%**")
        else:
            st.warning("⚠️ Image already optimized")

        # =================================================
        # DOWNLOAD BUTTON
        # =================================================
        st.download_button(
            f"⬇️ Download {uploaded_file.name}",
            compressed_bytes,
            file_name=f"compressed_{uploaded_file.name}",
            mime="image/jpeg"
        )

        st.markdown("---")

else:
    st.info("⬆️ Upload one or more images to start compression")