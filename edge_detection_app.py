import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Edge Detection App", layout="wide")

st.title("Edge Detection Web App")
st.markdown("**Sobel · Laplacian · Canny** — Upload an image and apply classical edge detection operators")

st.sidebar.header("Parameters")

operator=st.sidebar.selectbox(
    "Select Operator",
    ["All Three (Side-by-Side)","Sobel","Laplacian","Canny"]
)

sobel_ksize=st.sidebar.slider("Sobel Kernel Size (odd)",1,7,5,step=2)
lap_ksize=st.sidebar.slider("Laplacian Kernel Size (odd)",1,7,3,step=2)
canny_low=st.sidebar.slider("Canny Low Threshold",10,200,100)
canny_high=st.sidebar.slider("Canny High Threshold",50,400,200)

uploaded=st.file_uploader("Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"])
url_input = st.text_input("Or paste an image URL:")

if url_input:
    import requests
    from io import BytesIO
    response=requests.get(url_input)
    uploaded=BytesIO(response.content)

if uploaded is not None:
    pil_img=Image.open(uploaded).convert("RGB")
    img_rgb=np.array(pil_img)
    img_bgr=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def recommend_operator(gray):
        blur_score=cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast=gray.std()
        if blur_score<400:
            return "Canny","Image appears blurry or noisy — Canny handles this best with its built-in Gaussian pre-blur."
        elif contrast>60:
            return "Sobel","High contrast image — Sobel captures strong gradient edges well."
        else:
            return "Laplacian","Moderate contrast — Laplacian picks up fine detail edges."

    rec_op,rec_reason=recommend_operator(gray)
    st.info(f"**Recommended Operator: {rec_op}** — {rec_reason}")
    def compute_sobel(gray,ksize):
        gx=cv2.Sobel(gray,cv2.CV_64F,dx=1,dy=0,ksize=ksize)
        gy=cv2.Sobel(gray,cv2.CV_64F,dx=0,dy=1,ksize=ksize)
        result=np.sqrt(gx**2 + gy**2)
        return cv2.convertScaleAbs(result)

    def compute_laplacian(gray,ksize):
        result=cv2.Laplacian(gray, cv2.CV_64F,ksize=ksize)
        return cv2.convertScaleAbs(result)

    def compute_canny(gray,low,high):
        return cv2.Canny(gray,low,high)
    
    if operator=="All Three (Side-by-Side)":
        st.subheader("Results")
        col1,col2,col3,col4=st.columns(4)
        sobel_out=compute_sobel(gray,sobel_ksize)
        lap_out=compute_laplacian(gray,lap_ksize)
        canny_out=compute_canny(gray,canny_low,canny_high)

        with col1:
            st.image(img_rgb,caption="Original",use_container_width=True)
        with col2:
            st.image(sobel_out,caption=f"Sobel",use_container_width=True,clamp=True)
        with col3:
            st.image(lap_out,caption="Laplacian",use_container_width=True,clamp=True)
        with col4:
            st.image(canny_out,caption=f"Canny (low={canny_low}, high={canny_high})",use_container_width=True,clamp=True)

    elif operator=="Sobel":
        st.subheader("Sobel Edge Detection")
        sobel_out=compute_sobel(gray,sobel_ksize)
        col1, col2=st.columns(2)
        with col1:
            st.image(img_rgb,caption="Original",use_container_width=True)
        with col2:
            st.image(sobel_out,caption=f"Sobel — ksize={sobel_ksize}",use_container_width=True,clamp=True)

    elif operator=="Laplacian":
        st.subheader("Laplacian Edge Detection")
        lap_out=compute_laplacian(gray,lap_ksize)
        col1,col2=st.columns(2)
        with col1:
            st.image(img_rgb, caption="Original", use_container_width=True)
        with col2:
            st.image(lap_out, caption=f"Laplacian — ksize={lap_ksize}", use_container_width=True, clamp=True)

    elif operator=="Canny":
        st.subheader("Canny Edge Detection")
        canny_out=compute_canny(gray, canny_low, canny_high)
        col1, col2=st.columns(2)
        with col1:
            st.image(img_rgb,caption="Original",use_container_width=True)
        with col2:
            st.image(canny_out,caption=f"Canny — low={canny_low}, high={canny_high}",use_container_width=True,clamp=True)

else:
    st.info("Upload an image using the panel above to get started.")
    st.markdown("""
    **How to use:**
    1. Upload any JPG or PNG image
    2. Pick an operator from the sidebar (or view all three at once)
    3. Adjust the parameters using the sliders
    4. Results update instantly
    """)
