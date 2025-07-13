import streamlit as st
import gdown
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

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
    
    # Load detection model
    det_model = YOLO("best.pt")
    
    # Load type classification model
    type_classes = ['Bus', 'Car', 'Motorcycle', 'Truck']
    type_model = resnet18(pretrained=False)
    type_model.fc = nn.Linear(type_model.fc.in_features, len(type_classes))
    type_model.load_state_dict(torch.load("resnet18_vehicle_cls.pth", map_location=device))
    type_model.to(device).eval()
    
    # Load color classification model
    color_classes = [
        'Bus_black', 'Bus_silver', 'Car_black', 'Car_blue', 'Car_red', 'Car_silver', 
        'Car_white', 'Car_yellow', 'Motorcycle_black', 'Motorcycle_red', 'Motorcycle_silver', 
        'Motorcycle_white', 'Motorcycle_yellow', 'Truck_black', 'Truck_red', 'Truck_silver', 
        'Truck_white', 'Extra_class_1', 'Extra_class_2'
    ]
    color_model = resnet18(pretrained=False)
    color_model.fc = nn.Linear(color_model.fc.in_features, len(color_classes))
    color_model.load_state_dict(torch.load("resnet18_color_subclass.pth", map_location=device))
    color_model.to(device).eval()
    
    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return det_model, type_model, color_model, type_classes, color_classes, transform, device

# ================================
# Enhanced Visualization Functions
# ================================
def draw_detections(image, detections):
    """Draw bounding boxes with color-coded labels"""
    draw = ImageDraw.Draw(image)
    font_size = max(12, int(image.width / 50))
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        label = f"{detection['type']} - {detection['color']}"
        
        # Draw bounding box with color matching vehicle color
        box_color = COLOR_MAP.get(detection['color'].lower(), "#FF0000")
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1), label)
        text_width = text_bbox[2] - text_bbox[0] + 10
        text_height = text_bbox[3] - text_bbox[1] + 5
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="black")
        
        # Draw text
        draw.text((x1 + 5, y1 - text_height + 2), label, fill="white")
    
    return image

def create_analytics_charts(type_counts, color_counts):
    """Create separate charts for vehicle types and colors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Vehicle type distribution
    types = list(type_counts.keys())
    counts = list(type_counts.values())
    ax1.bar(types, counts, color='#4C72B0')
    ax1.set_title('Vehicle Type Distribution', fontsize=14)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Color distribution with actual colors
    colors = list(color_counts.keys())
    color_values = [COLOR_MAP.get(c.lower(), "#999999") for c in colors]
    counts = list(color_counts.values())
    
    ax2.bar(colors, counts, color=color_values)
    ax2.set_title('Color Distribution', fontsize=14)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# ================================
# Enhanced Inference Function
# ================================
def run_enhanced_inference(image, det_model, type_model, color_model, 
                          type_classes, color_classes, transform, device):
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    results = det_model(img_bgr)
    boxes = results[0].boxes
    
    detections = []
    type_counts = defaultdict(int)
    color_counts = defaultdict(int)
    
    for box in boxes:
        # Extract bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_np[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop)
        
        # Classify vehicle type
        input_tensor = transform(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            type_pred = type_model(input_tensor)
            type_label = type_classes[type_pred.argmax().item()]
            
            # Classify color
            color_pred = color_model(input_tensor)
            color_label = color_classes[color_pred.argmax().item()]
            vehicle_color = color_label.split('_')[1] if '_' in color_label else color_label
        
        # Store detection details
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "type": type_label,
            "color": vehicle_color,
            "confidence": float(box.conf.item())
        })
        
        # Update counts
        type_counts[type_label] += 1
        color_counts[vehicle_color] += 1
    
    # Convert to regular dicts
    type_counts = dict(type_counts)
    color_counts = dict(color_counts)
    
    return detections, type_counts, color_counts

# ================================
# Streamlit UI
# ================================
# Color mapping for visualization
COLOR_MAP = {
    "red": "#FF0000", "blue": "#0000FF", "green": "#00FF00", 
    "yellow": "#FFFF00", "black": "#000000", "white": "#FFFFFF",
    "silver": "#C0C0C0", "gray": "#808080", "brown": "#A52A2A"
}

# Vehicle emojis for display
VEHICLE_EMOJIS = {
    "Bus": "🚌", "Car": "🚗", "Motorcycle": "🏍️", "Truck": "🚚"
}

def main():
    # Configure page
    st.set_page_config(
        page_title="Vehicle Detection System", 
        page_icon="🚘", 
        layout="wide"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    .stMarkdown h1 { color: #2e3a59; }
    .stButton>button { background-color: #4c72b0; color: white; }
    .stFileUploader>div>div { border: 2px dashed #4c72b0; }
    .summary-card { 
        background: white; 
        border-radius: 10px; 
        padding: 15px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-value { 
        font-size: 1.8rem; 
        font-weight: bold; 
        color: #4c72b0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Page header
    st.title("🚘 Advanced Vehicle Detection System")
    st.markdown("""
    Upload an image to detect vehicles, classify their type, and identify their color.
    Powered by YOLOv8 + ResNet18 models.
    """)
    
    # Download & load models
    with st.spinner("Loading AI models..."):
        download_models()
        models = load_models()
        det_model, type_model, color_model, type_classes, color_classes, transform, device = models
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with st.spinner("Detecting and classifying vehicles..."):
            detections, type_counts, color_counts = run_enhanced_inference(
                image, det_model, type_model, color_model,
                type_classes, color_classes, transform, device
            )
        
        if detections:
            # Create processed image
            processed_img = draw_detections(image.copy(), detections)
            
            with col2:
                st.subheader("Detection Results")
                st.image(processed_img, use_column_width=True)
                
                # Summary statistics
                total_count = len(detections)
                st.success(f"✅ Detected {total_count} vehicles")
                
                # Type summary with emojis
                type_summary = " | ".join(
                    [f"{VEHICLE_EMOJIS.get(k, '🚙')} {k}: {v}" 
                     for k, v in type_counts.items()]
                )
                st.markdown(f"**Vehicle Types:** {type_summary}")
                
                # Color summary with color coding
                color_summary = " | ".join(
                    [f"<span style='color:{COLOR_MAP.get(c.lower(), '#000000')}'>{c}</span>: {v}" 
                     for c, v in color_counts.items()]
                )
                st.markdown(f"**Vehicle Colors:** {color_summary}", unsafe_allow_html=True)
            
            # Analytics section
            st.divider()
            st.subheader("Analytics Dashboard")
            
            # Create analytics charts
            fig = create_analytics_charts(type_counts, color_counts)
            st.pyplot(fig)
            
            # Detection details table
            st.subheader("Detection Details")
            df = pd.DataFrame(detections)
            df = df[['type', 'color', 'confidence']]
            
            # Apply color coding to table
            st.dataframe(
                df.style.applymap(
                    lambda x: f"background-color: {COLOR_MAP.get(x.lower(), '#FFFFFF')}" 
                    if x in COLOR_MAP else '',
                    subset=['color']
                ).format({"confidence": "{:.2f}"}),
                height=400
            )
        else:
            st.warning("⚠️ No vehicles detected. Please try another image.")
    
    # Sidebar information
    with st.sidebar:
        st.header("System Information")
        st.markdown("""
        **Models:**
        - Detection: YOLOv8
        - Classification: ResNet18
        - Color: ResNet18
        
        **Vehicle Types:**
        - Bus, Car, Motorcycle, Truck
        
        **Vehicle Colors:**
        - Black, Blue, Red, Silver, White, Yellow
        """)
        st.info("For best results, use clear images of vehicles")

if __name__ == "__main__":
    main()
