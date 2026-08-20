import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import fitz  # PyMuPDF
from moviepy import VideoFileClip

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Universal Media Compressor",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Universal Media Compressor")
st.caption("Compress Images • Videos • PDFs")

# ======================================================
# FILE UPLOADER
# ======================================================
uploaded_files = st.file_uploader(
    "Upload files",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "pdf"],
    accept_multiple_files=True
)

# ======================================================
# MAIN LOGIC
# ======================================================
if uploaded_files:

    for uploaded_file in uploaded_files:

        st.markdown(f"## 📄 {uploaded_file.name}")
        file_type = uploaded_file.name.split(".")[-1].lower()

        # ======================================================
        # 🖼 IMAGE COMPRESSION
        # ======================================================
        if file_type in ["jpg", "jpeg", "png"]:

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            original_size = uploaded_file.size / 1024

            # Compression
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            success, encoded_img = cv2.imencode(".jpg", image_bgr, encode_param)

            compressed_bytes = encoded_img.tobytes()
            final_size = len(compressed_bytes) / 1024

            compressed_np = np.frombuffer(compressed_bytes, np.uint8)
            compressed_bgr = cv2.imdecode(compressed_np, cv2.IMREAD_COLOR)
            compressed_rgb = cv2.cvtColor(compressed_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Original")
                st.image(image_rgb, use_container_width=True)
                st.write(f"Size: {original_size:.2f} KB")

            with col2:
                st.subheader("🗜️ Compressed")
                st.image(compressed_rgb, use_container_width=True)
                st.write(f"Size: {final_size:.2f} KB")

            if final_size >= original_size:
                st.warning("⚠ Image already optimized. Compression may increase size.")

            st.download_button(
                "⬇ Download Image",
                compressed_bytes,
                file_name=f"compressed_{uploaded_file.name}",
                mime="image/jpeg"
            )

        # ======================================================
        # 🎥 VIDEO COMPRESSION
        # ======================================================
        elif file_type in ["mp4", "mov", "avi"]:

            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(uploaded_file.read())

            st.video(temp_video.name)

            if st.button(f"Compress {uploaded_file.name}"):

                with st.spinner("Compressing video..."):

                    clip = VideoFileClip(temp_video.name)

                    # ✅ FIXED VERSION (works for you)
                    clip_resized = clip.resized(width=clip.w // 2)

                    output_path = f"compressed_{uploaded_file.name}"

                    clip_resized.write_videofile(
                        output_path,
                        bitrate="300k",
                        codec="libx264",
                        audio_codec="aac"
                    )

                    original_size = os.path.getsize(temp_video.name) / 1024 / 1024
                    compressed_size = os.path.getsize(output_path) / 1024 / 1024

                st.success("Video compressed successfully!")

                st.write(f"Original: {original_size:.2f} MB")
                st.write(f"Compressed: {compressed_size:.2f} MB")

                if compressed_size >= original_size:
                    st.warning("⚠ Video already optimized. Compression may increase size.")

                with open(output_path, "rb") as f:
                    st.download_button(
                        "⬇ Download Video",
                        f,
                        file_name=output_path
                    )

        # ======================================================
        # 📄 PDF COMPRESSION
        # ======================================================
        elif file_type == "pdf":

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                input_path = tmp.name

            doc = fitz.open(input_path)
            output_pdf = f"compressed_{uploaded_file.name}"

            doc.save(
                output_pdf,
                garbage=4,
                deflate=True
            )

            original_size = uploaded_file.size / 1024
            final_size = os.path.getsize(output_pdf) / 1024

            st.write(f"Original: {original_size:.2f} KB")
            st.write(f"Compressed: {final_size:.2f} KB")

            if final_size >= original_size:
                st.warning("⚠ PDF already optimized. Compression may not reduce size.")

            with open(output_pdf, "rb") as f:
                st.download_button(
                    "⬇ Download PDF",
                    f,
                    file_name=output_pdf
                )

        st.divider()

else:
    st.info("Upload images, videos, or PDFs to start compression.")