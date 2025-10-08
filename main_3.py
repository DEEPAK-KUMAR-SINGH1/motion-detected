import streamlit as st
import cv2
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import numpy as np

st.title("🤖 YOLOS Live Object Detection (Streamlit)")

# Load YOLOS model
@st.cache_resource
def load_model():
    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device

processor, model, device = load_model()

# Video capture
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{model.config.id2label[label.item()]} {score:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert BGR to RGB and display in Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

st.text("Press the checkbox to start/stop the camera")
