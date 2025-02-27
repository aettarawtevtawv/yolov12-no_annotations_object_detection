import os
import torch
from transformers import DetrImageProcessor
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import threading
import queue
import time
import logging
from ultralytics import YOLO

# Initialize logging
logging.basicConfig(filename='detections.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Load the object detection model (YOLOv12x)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device: {device}")  # Verify GPU usage
try:
    yolo_model = YOLO("yolov12x.pt")  # Load yolov12x.pt - MAKE SURE THIS FILE EXISTS!
except Exception as e:
    logging.error(f"Error loading yolov12x.pt: {e}.  Ensure yolov12x.pt is in the script's directory or provide the correct path.")
    print(f"Error loading yolov12x.pt: {e}. Please check the log file and model file path.")
    exit()  # Exit if yolov12x.pt cannot be loaded
yolo_model.to(device)
yolo_model.eval()

# Load the CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Path to the reference images
reference_image_dir = 'reference_images/'

# Function to load reference images and extract their features
def load_reference_images():
    reference_images = []
    labels = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for category in os.listdir(reference_image_dir):
        category_dir = os.path.join(reference_image_dir, category)
        if os.path.isdir(category_dir):
            for img_name in os.listdir(category_dir):
                img_path = os.path.join(category_dir, img_name)
                if os.path.splitext(img_path)[1].lower() not in valid_extensions:
                    continue
                try:
                    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        img_features = clip_model.encode_image(img)
                        img_features /= img_features.norm(dim=-1, keepdim=True)
                    reference_images.append(img_features)
                    labels.append(category)
                except UnidentifiedImageError:
                    logging.warning(f"Skipping non-image file: {img_path}")
                except Exception as e:
                    logging.error(f"Error processing image {img_path}: {e}")
    return torch.cat(reference_images), labels

# Load reference images and their features
reference_features, reference_labels = load_reference_images()

def detect_objects(img_pil):
    try:
        # Resize the image - smaller size for potential speedup (optional, experiment)
        img_pil = img_pil.resize((320, 240))  # Reduced resize for potentially faster processing
        frame_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Perform object detection with YOLOv12x
        results = yolo_model(frame_np, conf=0.45)  # Confidence threshold (adjust if needed)

        boxes = []
        scores = []
        for result in results:
            if result is not None and hasattr(result, 'boxes') and result.boxes is not None:
                numpy_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                numpy_scores = result.boxes.conf.cpu().numpy()
                boxes.extend(numpy_boxes)
                scores.extend(numpy_scores)
        return boxes, scores
    except Exception as e:
        logging.error(f"Error in detect_objects: {e}")
        return [], []

def classify_objects_clip(img_crops):
    results = []
    try:
        img_preprocessed = torch.stack([preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))) for crop in img_crops]).to(device)

        with torch.no_grad():
            image_embeddings = clip_model.encode_image(img_preprocessed)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

            # Compute similarities and probabilities
            similarities = image_embeddings @ reference_features.T
            best_match_indices = similarities.argmax(dim=-1).cpu().numpy()
            best_confidences = similarities.max(dim=-1)[0].cpu().numpy()

        for idx in range(len(img_crops)):
            best_match_idx = best_match_indices[idx]
            confidence = best_confidences[idx]
            if confidence >= 0.45:
                best_match_label = reference_labels[best_match_idx]
                results.append((best_match_label, confidence))
            else:
                results.append(("unknown", confidence))
    except Exception as e:
        logging.error(f"Error in classify_objects_clip: {e}")
        results = [("unknown", 0.0) for _ in img_crops]
    return results

def video_capture(frame_queue):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            logging.info(f"Using video capture backend: {backend}")
            break

    if not cap or not cap.isOpened():
        logging.error("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Failed to read frame from webcam.")
            break
        frame_queue.put(frame)
        time.sleep(0.01)  # Adjust sleep time as needed

    cap.release()

def main():
    frame_queue = queue.Queue()
    threading.Thread(target=video_capture, args=(frame_queue,), daemon=True).start()
    frame_count = 0 # For optional frame skipping

    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        frame_count += 1
        process_every_n_frames = 1  # Process every frame initially - adjust for skipping if needed
        if frame_count % process_every_n_frames != 0:
            continue # Skip frames if process_every_n_frames > 1

        try:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, scores = detect_objects(img_pil)

            img_crops = []
            box_coords = []
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(frame.shape[1], x_max), min(frame.shape[0], y_max)
                crop = frame[y_min:y_max, x_min:x_max]
                img_crops.append(crop)
                box_coords.append((x_min, y_min, x_max, y_max))

            if img_crops:
                clip_results = classify_objects_clip(img_crops)

                for (x_min, y_min, x_max, y_max), (label, confidence), score in zip(box_coords, clip_results, scores):
                    color = (0, 255, 0) if label != "unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    label_text = f"{label} ({confidence*100:.1f}%) | Detected ({score*100:.1f}%)"
                    (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x_min, y_min - label_height - 10), (x_min + label_width, y_min), color, cv2.FILLED)
                    cv2.putText(frame, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    logging.info(f"Detected: {label} with confidence {confidence*100:.1f}% at position {(x_min, y_min, x_max, y_max)}")

            cv2.namedWindow('Advanced Object Detection', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Advanced Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Advanced Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            continue

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()