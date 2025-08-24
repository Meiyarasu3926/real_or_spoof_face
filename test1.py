import streamlit as st
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ExifTags
import cv2
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Fix for PyTorch 2.6 weights_only security update
try:
    import torch.serialization
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
except:
    pass  # For older PyTorch versions

def preprocess_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Preprocess image for consistent face detection with proper orientation"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Handle EXIF orientation
    try:
        if hasattr(image, '_getexif') and image._getexif():
            exif = image._getexif()
            for tag, name in ExifTags.TAGS.items():
                if name == 'Orientation' and tag in exif:
                    orientation = exif[tag]
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
                    break
    except Exception as e:
        logger.warning(f"Could not process EXIF data: {e}")
    
    # Resize large images while maintaining aspect ratio
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

def enhance_image_for_detection(image: Image.Image) -> Image.Image:
    """Enhance image contrast and brightness for better face detection"""
    img_array = np.array(image)
    
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(enhanced)
class ProductionAntiSpoofModel(nn.Module):
    def __init__(self, pretrained=False):
        super(ProductionAntiSpoofModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Keep more layers trainable for better performance
        # Only freeze very early layers
        for param in list(self.backbone.parameters())[:30]:
            param.requires_grad = False
        
        # Production classifier with balanced regularization
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Moderate dropout
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # Real=1, Spoof=0
        )
    
    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_models():
    """Load models and cache them"""
    # Load YOLOv8 face detection model
    try:
        face_model = YOLO('yolov8n-face.pt')
        st.success("✅ YOLOv8 face detection model loaded!")
    except Exception as e:
        st.error(f"❌ YOLOv8 face model error: {str(e)}")
        st.info("Trying alternative face detection models...")
        
        # Try alternative model names
        alternative_models = ['yolo11n-face.pt', 'yolov8n.pt', 'yolov11n.pt']
        face_model = None
        
        for model_name in alternative_models:
            try:
                face_model = YOLO(model_name)
                st.success(f"✅ Loaded alternative model: {model_name}")
                break
            except:
                continue
        
        if face_model is None:
            st.error("❌ Could not load any face detection model!")
            return None, None
    
    # Load production anti-spoofing model
    spoof_model = ProductionAntiSpoofModel(pretrained=False)
    
    try:
        # Load the trained model weights with PyTorch 2.6 compatibility
        device = torch.device("cpu")  # Force CPU usage for deployment
        
        try:
            # First try with weights_only=False for trusted checkpoint
            checkpoint = torch.load('production_antispoofing_final.pth', 
                                  map_location=device, weights_only=False)
        except Exception as e:
            if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
                # Fallback: Try loading with older PyTorch method
                try:
                    checkpoint = torch.load('production_antispoofing_final.pth', 
                                          map_location=device)
                except:
                    st.error("❌ Unable to load model. Please re-save your model with current PyTorch version.")
                    return face_model, None
            else:
                raise e
        
        spoof_model.load_state_dict(checkpoint['model_state_dict'])
        spoof_model = spoof_model.to(device)
        spoof_model.eval()
        
        st.success("✅ Production Anti-Spoofing Model loaded successfully!")
        
        # Display model performance info
        if 'results' in checkpoint:
            results = checkpoint['results']
            st.info(f"📊 Model Performance: "
                   f"Train Acc: {results['train_acc']:.1f}%, "
                   f"Val Acc: {results['val_acc']:.1f}%, "
                   f"AUC: {results['auc']:.3f}")
        
    except FileNotFoundError:
        st.error("❌ Production model 'production_antispoofing_final.pth' not found!")
        st.info("Please ensure the trained model file is in the current directory.")
        return face_model, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return face_model, None
    
    return face_model, spoof_model

def predict_antispoofing(face_image, model, device, confidence_threshold=0.8):
    """Predict if face is real or spoof with confidence"""
    # Production transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare input
    input_tensor = transform(face_image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence = torch.max(probabilities, dim=1)[0].item()
        predicted = torch.argmax(outputs, dim=1).item()
    
    # Convert prediction (Real=1, Spoof=0)
    is_real = predicted == 1
    real_confidence = probabilities[0][1].item()
    spoof_confidence = probabilities[0][0].item()
    
    # Determine result based on confidence threshold
    if confidence < confidence_threshold:
        result = "UNCERTAIN"
        status_color = "🟡"
    else:
        result = "REAL" if is_real else "SPOOF"
        status_color = "🟢" if is_real else "🔴"
    
    return {
        'result': result,
        'confidence': confidence,
        'real_confidence': real_confidence,
        'spoof_confidence': spoof_confidence,
        'status_color': status_color
    }

def crop_face_with_padding(img_array, box, padding_factor=0.2):
    """Crop face with padding for better context"""
    x1, y1, x2, y2 = map(int, box[:4])
    
    # Calculate padding
    width = x2 - x1
    height = y2 - y1
    pad_w = int(width * padding_factor)
    pad_h = int(height * padding_factor)
    
    # Apply padding with bounds checking
    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(img_array.shape[1], x2 + pad_w)
    y2_pad = min(img_array.shape[0], y2 + pad_h)
    
    # Crop face
    face = img_array[y1_pad:y2_pad, x1_pad:x2_pad]
    return face, (x1_pad, y1_pad, x2_pad, y2_pad)

# Streamlit App
def main():
    st.set_page_config(
        page_title="🛡️ Anti-Spoofing System", 
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title('🛡️ Production Anti-Spoofing System')
    st.markdown("""
    **Real-time Face Liveness Detection** - Distinguishes between live faces and screen/photo displays
    
    Upload an image to detect faces and classify them as **REAL** (live person) or **SPOOF** (screen/photo).
    """)
    
    # Load models
    face_model, spoof_model = load_models()
    
    if spoof_model is None:
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.5, 
        max_value=1.0, 
        value=0.8, 
        step=0.05,
        help="Minimum confidence for classification. Lower values = more sensitive."
    )
    
    show_face_crops = st.sidebar.checkbox("Show Individual Face Crops", value=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing faces to analyze"
    )
    
    if uploaded_file is not None:
        # Load and preprocess original image
        image = Image.open(uploaded_file)
        
        # Preprocess image for better face detection
        processed_image = preprocess_image(image, max_size=1024)
        enhanced_image = enhance_image_for_detection(processed_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Convert for processing
        img_array = np.array(enhanced_image)
        device = torch.device("cpu")
        
        # Face Detection with multiple attempts
        with st.spinner("🔍 Detecting faces..."):
            # Try detection on enhanced image first
            results = face_model(img_array, conf=0.3, iou=0.5)
            
            # If no faces found, try original image
            if not any(len(result.boxes) > 0 for result in results if hasattr(result, 'boxes')):
                st.info("🔄 No faces in enhanced image, trying original...")
                results = face_model(np.array(processed_image), conf=0.25, iou=0.5)
            
            # If still no faces, try with different parameters
            if not any(len(result.boxes) > 0 for result in results if hasattr(result, 'boxes')):
                st.info("🔄 Trying with relaxed detection parameters...")
                results = face_model(np.array(processed_image), conf=0.15, iou=0.7)
        
        # Process detection results
        detected_faces = []
        detection_img = img_array.copy()
        faces_found = False
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                faces_found = True
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else [1.0] * len(boxes)
                
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    # Crop face with padding
                    face_crop, crop_coords = crop_face_with_padding(img_array, box)
                    x1, y1, x2, y2 = crop_coords
                    
                    # Convert to PIL and resize
                    face_pil = Image.fromarray(face_crop)
                    face_resized = face_pil.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    detected_faces.append({
                        'image': face_resized,
                        'bbox': (x1, y1, x2, y2),
                        'original_bbox': box,
                        'confidence': conf
                    })
                    
                    # Draw bounding box on detection image
                    color = (0, 255, 0) if conf > 0.5 else (255, 255, 0)
                    thickness = 3 if conf > 0.5 else 2
                    cv2.rectangle(detection_img, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(detection_img, f'Face {i+1} ({conf:.2f})', (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        with col2:
            st.subheader("🎯 Face Detection Results")
            if faces_found and detected_faces:
                st.success(f"✅ Detected {len(detected_faces)} face(s)")
                st.image(detection_img, caption=f'Detected {len(detected_faces)} face(s)', 
                        use_container_width=True)
            else:
                st.warning("⚠️ No faces detected in the image")
                st.info("💡 Try uploading an image with:")
                st.write("• Clearer face visibility")
                st.write("• Better lighting")
                st.write("• Face size > 50x50 pixels")
                st.write("• Less rotation/tilt")
                st.image(img_array, caption='No faces found', use_container_width=True)
        
        # Anti-Spoofing Classification
        if detected_faces:
            st.header("🛡️ Anti-Spoofing Analysis")
            
            results_summary = []
            
            for i, face_data in enumerate(detected_faces):
                with st.spinner(f"🔍 Analyzing Face {i+1}..."):
                    prediction = predict_antispoofing(
                        face_data['image'], 
                        spoof_model, 
                        device, 
                        confidence_threshold
                    )
                
                results_summary.append(prediction)
                
                # Display results
                col_face, col_result = st.columns([1, 2])
                
                if show_face_crops:
                    with col_face:
                        st.image(face_data['image'], 
                                caption=f'Face {i+1} (224x224)', 
                                width=200)
                
                with col_result:
                    result_text = f"{prediction['status_color']} **Face {i+1}: {prediction['result']}**"
                    st.markdown(result_text, unsafe_allow_html=True)
                    
                    # Confidence metrics
                    st.metric(
                        label="Overall Confidence", 
                        value=f"{prediction['confidence']:.1%}",
                        delta=None
                    )
                    
                    # Detailed probabilities
                    col_real, col_spoof = st.columns(2)
                    with col_real:
                        st.metric("Real Probability", f"{prediction['real_confidence']:.1%}")
                    with col_spoof:
                        st.metric("Spoof Probability", f"{prediction['spoof_confidence']:.1%}")
                    
                    # Status interpretation
                    if prediction['result'] == "REAL":
                        st.success("✅ Live person detected - Authentication recommended")
                    elif prediction['result'] == "SPOOF":
                        st.error("❌ Spoof detected - Reject authentication")
                    else:
                        st.warning("⚠️ Uncertain result - Manual review recommended")
                
                st.divider()
            
            # Summary Statistics
            st.header("📊 Detection Summary")
            
            real_count = sum(1 for r in results_summary if r['result'] == 'REAL')
            spoof_count = sum(1 for r in results_summary if r['result'] == 'SPOOF')
            uncertain_count = sum(1 for r in results_summary if r['result'] == 'UNCERTAIN')
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Faces", len(detected_faces))
            with col2:
                st.metric("Real Faces", real_count, delta_color="normal")
            with col3:
                st.metric("Spoof Faces", spoof_count, delta_color="inverse")
            with col4:
                st.metric("Uncertain", uncertain_count, delta_color="off")
            
            # Overall recommendation
            if spoof_count > 0:
                st.error("🚨 **SECURITY ALERT**: Spoof face(s) detected - Reject authentication")
            elif uncertain_count > 0:
                st.warning("⚠️ **CAUTION**: Uncertain detection(s) - Manual review recommended")
            else:
                st.success("✅ **ALL CLEAR**: Only live faces detected - Proceed with authentication")
        
        # Technical Details (Expandable)
        with st.expander("🔬 Technical Details"):
            st.markdown("""
            **Model Architecture**: EfficientNet-B0 with custom classifier
            **Input Size**: 224x224 pixels
            **Classes**: Real (live person) vs Spoof (screen/photo display)
            **Training Performance**: 100% validation accuracy, AUC=1.000
            
            **Detection Process**:
            1. YOLOv8 face detection with bounding boxes
            2. Face cropping with 20% padding for context
            3. Resize to 224x224 pixels
            4. Normalize using ImageNet statistics
            5. Anti-spoofing classification with confidence scoring
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("🔐 **Secure Face Liveness Detection** | Built with Streamlit & PyTorch")

if __name__ == "__main__":
    main()