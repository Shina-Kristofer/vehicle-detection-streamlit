
import sys
if sys.version_info >= (3, 13):
    st.error("Python 3.13 is not supported. Please use Python 3.10")
    st.stop()
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
import plotly.graph_objects as go
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

def create_3d_bar_chart(type_counts, color_counts):
    """Create 3D bar charts for vehicle types and colors"""
    # Vehicle type distribution
    fig1 = go.Figure(data=[go.Bar3d(
        x=list(type_counts.keys()),
        y=[0] * len(type_counts),
        z=list(type_counts.values()),
        colorscale='Viridis',
        opacity=0.8
    ])
    
    fig1.update_layout(
        title='<b>3D Vehicle Type Distribution</b>',
        scene=dict(
            xaxis_title='Vehicle Type',
            yaxis_title='',
            zaxis_title='Count',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # Color distribution
    fig2 = go.Figure(data=[go.Bar3d(
        x=list(color_counts.keys()),
        y=[0] * len(color_counts),
        z=list(color_counts.values()),
        colorscale=[[0, COLOR_MAP.get(c.lower(), "#999999")] for c in color_counts.keys()],
        opacity=0.8
    ])
    
    fig2.update_layout(
        title='<b>3D Color Distribution</b>',
        scene=dict(
            xaxis_title='Vehicle Color',
            yaxis_title='',
            zaxis_title='Count',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig1, fig2

def create_3d_surface_chart(cross_df):
    """Create 3D surface chart for vehicle type vs color distribution"""
    types = cross_df.index.tolist()
    colors = cross_df.columns.tolist()
    
    fig = go.Figure(data=[go.Surface(
        z=cross_df.values,
        x=types,
        y=colors,
        colorscale='Viridis',
        opacity=0.9,
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    ])
    
    fig.update_layout(
        title='<b>Vehicle Type vs Color Distribution</b>',
        scene=dict(
            xaxis_title='Vehicle Type',
            yaxis_title='Color',
            zaxis_title='Count',
            camera=dict(eye=dict(x=1.7, y=1.7, z=1))
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
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
    
    # Create cross-tabulation dataframe
    df = pd.DataFrame(detections)
    if not df.empty:
        cross_df = pd.crosstab(df['type'], df['color'])
    else:
        cross_df = pd.DataFrame()
    
    return detections, type_counts, color_counts, cross_df

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
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark theme
    st.markdown("""
    <style>
    :root {
        --primary: #1e2130;
        --secondary: #2d3149;
        --accent: #4cc9f0;
        --text: #ffffff;
        --border: #4a4e69;
        --card: #2d3149;
        --success: #4ade80;
        --warning: #facc15;
        --danger: #f87171;
    }
    
    body {
        background-color: var(--primary) !important;
        color: var(--text) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--primary) 0%, #0f111a 100%);
        color: var(--text);
    }
    
    .stButton>button {
        background: var(--accent) !important;
        color: var(--primary) !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(76, 201, 240, 0.3);
    }
    
    .stFileUploader>div>div {
        background: var(--secondary) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 12px;
    }
    
    .metric-card {
        background: var(--card);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        border-left: 4px solid var(--accent);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
    }
    
    .metric-title {
        font-size: 1rem;
        font-weight: 500;
        color: #a0aec0;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent);
        margin: 0;
    }
    
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
        background: var(--secondary) !important;
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
    }
    
    .stSpinner>div {
        color: var(--accent) !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text) !important;
        border-bottom: 2px solid var(--accent);
        padding-bottom: 0.3em;
    }
    
    .stAlert {
        border-radius: 12px !important;
    }
    
    .stSuccess {
        background-color: rgba(76, 222, 128, 0.15) !important;
        border: 1px solid var(--success) !important;
    }
    
    .stWarning {
        background-color: rgba(250, 204, 21, 0.15) !important;
        border: 1px solid var(--warning) !important;
    }
    
    .stException {
        background-color: rgba(248, 113, 113, 0.15) !important;
        border: 1px solid var(--danger) !important;
    }
    
    .css-1d391kg { /* Sidebar */
        background: var(--secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Page header with gradient
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #1e2130 0%, #2d3149 50%, #1e2130 100%);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        border-left: 5px solid #4cc9f0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    ">
        <h1 style="color: #ffffff; margin: 0;">🚘 VISIONAI - Vehicle Detection System</h1>
        <p style="color: #a0aec0; margin: 0.5rem 0 0;">
        Advanced AI-powered vehicle detection, classification, and color recognition
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download & load models
    with st.spinner("🔍 Loading AI models..."):
        download_models()
        models = load_models()
        det_model, type_model, color_model, type_classes, color_classes, transform, device = models
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Traffic Image", type=["jpg", "jpeg", "png"], 
                                     help="Upload a clear image containing vehicles")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, use_column_width=True)
        
        with st.spinner("🚀 Detecting and classifying vehicles..."):
            detections, type_counts, color_counts, cross_df = run_enhanced_inference(
                image, det_model, type_model, color_model,
                type_classes, color_classes, transform, device
            )
        
        if detections:
            # Create processed image
            processed_img = draw_detections(image.copy(), detections)
            
            with col2:
                st.subheader("🔍 Detection Results")
                st.image(processed_img, use_column_width=True)
                
                # Summary statistics
                total_count = len(detections)
                st.success(f"✅ Detected {total_count} vehicles")
                
                # Metrics cards
                col_metrics = st.columns(3)
                with col_metrics[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">TOTAL VEHICLES</div>
                        <div class="metric-value">{total_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Find most common type and color
                most_common_type = max(type_counts, key=type_counts.get) if type_counts else "N/A"
                most_common_color = max(color_counts, key=color_counts.get) if color_counts else "N/A"
                
                with col_metrics[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">MOST COMMON TYPE</div>
                        <div class="metric-value">{VEHICLE_EMOJIS.get(most_common_type, '🚙')} {most_common_type}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_metrics[2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">MOST COMMON COLOR</div>
                        <div class="metric-value" style="color: {COLOR_MAP.get(most_common_color.lower(), '#FFFFFF')}">
                            {most_common_color.capitalize()}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Analytics section
            st.divider()
            st.subheader("📊 Advanced Analytics")
            
            # Create 3D visualizations
            if not cross_df.empty:
                # 3D Surface Chart
                st.markdown("### 🌐 Vehicle Type vs Color Distribution")
                surface_fig = create_3d_surface_chart(cross_df)
                st.plotly_chart(surface_fig, use_container_width=True)
                
                # 3D Bar Charts
                col_3d1, col_3d2 = st.columns(2)
                with col_3d1:
                    st.markdown("### 🚗 Vehicle Type Distribution")
                    bar3d_type, _ = create_3d_bar_chart(type_counts, {})
                    st.plotly_chart(bar3d_type, use_container_width=True)
                
                with col_3d2:
                    st.markdown("### 🎨 Vehicle Color Distribution")
                    _, bar3d_color = create_3d_bar_chart({}, color_counts)
                    st.plotly_chart(bar3d_color, use_container_width=True)
            
            # Detection details table
            st.subheader("🔬 Detection Details")
            df = pd.DataFrame(detections)
            df = df[['type', 'color', 'confidence']]
            df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2f}")
            
            # Apply color coding to table
            st.dataframe(
                df.style.apply(
                    lambda x: [f"background: {COLOR_MAP.get(x['color'].lower(), '#FFFFFF')}" 
                              if x.name == 'color' else '' for i in x], 
                    axis=1
                ).format(precision=2),
                height=400,
                use_container_width=True
            )
        else:
            st.warning("⚠️ No vehicles detected. Please try another image.")
    
    # Sidebar information
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #2d3149 0%, #1e2130 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid #4cc9f0;
        ">
            <h3 style="color: #ffffff; margin-top: 0;">System Information</h3>
            <p style="color: #a0aec0;">
            <b>Models:</b><br>
            • Detection: YOLOv8<br>
            • Classification: ResNet18<br>
            • Color: ResNet18<br><br>
            
            <b>Vehicle Types:</b><br>
            • Bus, Car, Motorcycle, Truck<br><br>
            
            <b>Vehicle Colors:</b><br>
            • Black, Blue, Red, Silver, White, Yellow
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #2d3149 0%, #1e2130 100%);
            border-radius: 12px;
            padding: 1.5rem;
            border-left: 4px solid #4cc9f0;
        ">
            <h3 style="color: #ffffff; margin-top: 0;">Tips for Best Results</h3>
            <ul style="color: #a0aec0; padding-left: 1.2rem;">
                <li>Use clear, well-lit images</li>
                <li>Capture vehicles from side angles</li>
                <li>Avoid extreme weather conditions</li>
                <li>Ensure vehicles are not overlapping</li>
                <li>Higher resolution images work better</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
