
import streamlit as st
import gdown
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ================================
# Download Models from Google Drive
# ================================
@st.cache_resource
def download_models():
    gdown.download(id='19kwkHA5tvK5QQ0Fsd3mI6_gf6ED0NLnz', output='best.pt', quiet=False)
    gdown.download(id='12oAd6AF5YOHitewaoB_k5CNdL8BHasjT', output='resnet18_vehicle_cls.pth', quiet=False)
    gdown.download(id='1x3JpptPMTzCnnxa-mfb_xyUsHK8v4Cto', output='resnet18_color_subclass.pth', quiet=False)

# ================================
# Load Models
# ================================
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    det_model = YOLO("best.pt")

    type_classes = ['Bus', 'Car', 'Motorcycle', 'Truck']
    type_model = resnet18(pretrained=False)
    type_model.fc = nn.Linear(type_model.fc.in_features, len(type_classes))
    type_model.load_state_dict(torch.load("resnet18_vehicle_cls.pth", map_location=device))
    type_model.to(device).eval()

    color_classes = [
        'Bus_black', 'Bus_silver', 'Car_black', 'Car_blue', 'Car_red', 'Car_silver', 'Car_white', 'Car_yellow',
        'Motorcycle_black', 'Motorcycle_red', 'Motorcycle_silver', 'Motorcycle_white', 'Motorcycle_yellow',
        'Truck_black', 'Truck_red', 'Truck_silver', 'Truck_white', 'Extra_class_1', 'Extra_class_2'
    ]
    color_model = resnet18(pretrained=False)
    color_model.fc = nn.Linear(color_model.fc.in_features, len(color_classes))
    color_model.load_state_dict(torch.load("resnet18_color_subclass.pth", map_location=device))
    color_model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return det_model, type_model, color_model, type_classes, color_classes, transform, device

# ================================
# Inference
# ================================
def run_inference(image, det_model, type_model, color_model, type_classes, color_classes, transform, device):
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    results = det_model(img_bgr)
    boxes = results[0].boxes

    counts = {}
    label_list = []
    annotated_img = img_np.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_np[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop)
        input_tensor = transform(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            type_pred = type_model(input_tensor)
            type_label = type_classes[type_pred.argmax().item()]

            color_pred = color_model(input_tensor)
            color_label = color_classes[color_pred.argmax().item()]
            vehicle_color = color_label.split('_')[1]

        final_label = f"{type_label} - {vehicle_color}"
        label_list.append(final_label)

        # Draw on image
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated_img, final_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Count
        counts[type_label] = counts.get(type_label, 0) + 1
        counts[vehicle_color] = counts.get(vehicle_color, 0) + 1

    return annotated_img, label_list, counts

# ================================
# Streamlit UI
# ================================
def main():
    st.title("🚗 Vehicle Detection, Classification & Color Recognition")
    st.write("Upload an image to detect vehicles and classify their type and color.")

    # Step 1: Download & Load models
    download_models()
    det_model, type_model, color_model, type_classes, color_classes, transform, device = load_models()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Running detection and classification..."):
            output_img, labels, counts = run_inference(
                image, det_model, type_model, color_model,
                type_classes, color_classes, transform, device
            )

        st.image(output_img, caption="Detected Vehicles", use_column_width=True)

        st.subheader("Labels Detected:")
        st.write(", ".join(labels))

        st.subheader("Counts:")
        st.json(counts)

        st.subheader("Chart:")
        fig, ax = plt.subplots()
        ax.bar(counts.keys(), counts.values(), color='skyblue')
        ax.set_ylabel("Count")
        ax.set_title("Vehicle Types and Colors")
        plt.xticks(rotation=45)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
