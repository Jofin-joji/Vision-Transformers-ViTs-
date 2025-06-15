import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from fastervit import create_model
from collections import defaultdict, deque
import time

# --- Face Detector Choice ---
USE_MTCNN = False # <<< SET TO TRUE IF YOU INSTALL AND WANT TO USE MTCNN
                  # pip install facenet-pytorch
if USE_MTCNN:
    try:
        from facenet_pytorch import MTCNN
        print("MTCNN will be used for face detection.")
    except ImportError:
        print("facenet-pytorch not found. Falling back to Haar Cascade.")
        USE_MTCNN = False

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = 'fastervit_face_embeddings.pth'
KNOWN_FACES_DIR = 'known_faces_db'
EMBEDDINGS_FILE = 'known_face_embeddings.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_IMAGES_PER_PERSON = 10 # Consider increasing if accuracy is low
RESIZE_DIM = (224, 224)

# Transformation for inference
inference_transform = transforms.Compose([
    transforms.Resize(RESIZE_DIM),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Face Detector Initialization ---
face_cascade = None
mtcnn = None

if not USE_MTCNN:
    HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(HAARCASCADE_PATH):
        print(f"Error: Haar Cascade file not found at {HAARCASCADE_PATH}")
        exit()
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    print("Haar Cascade will be used for face detection.")
else:
    mtcnn = MTCNN(
        keep_all=True,        # Detect all faces
        device=DEVICE,
        min_face_size=40,     # Minimum face size to detect
        thresholds=[0.6, 0.7, 0.7], # Detection thresholds
        factor=0.709,         # Scale factor
        post_process=True,    # Apply post-processing (like landmark alignment)
        select_largest=False  # Don't only select the largest face if keep_all=True
    )
    if mtcnn is None: # Fallback if MTCNN failed to initialize
        print("MTCNN initialization failed. Falling back to Haar Cascade.")
        USE_MTCNN = False
        HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(HAARCASCADE_PATH): exit()
        face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)


# --- Flickering Reduction ---
RECOGNITION_HISTORY_SIZE = 5  # Number of past frames to consider for smoothing
CONFIDENCE_STABILITY_THRESHOLD = 3 # How many consecutive same predictions to be confident

# To store history for each tracked face (simplified tracking by bbox)
face_recognition_histories = {} # Key: face_id (e.g., from a tracker, or simplified for now)
next_face_id = 0


# ----------------------------
# Model Loading
# ----------------------------
def load_trained_model(model_path):
    print(f"Loading model from {model_path}...")
    model = create_model('faster_vit_0_224', pretrained=False)
    model.head = torch.nn.Identity()
    
    map_location = torch.device('cpu') if not torch.cuda.is_available() else None
    state_dict = torch.load(model_path, map_location=map_location)
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
    return model

# ----------------------------
# Embedding Extraction
# ----------------------------
def get_embedding(face_image_pil, model):
    if face_image_pil is None:
        return None
    # Ensure image is RGB
    if face_image_pil.mode != 'RGB':
        face_image_pil = face_image_pil.convert('RGB')
        
    transformed_face = inference_transform(face_image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(transformed_face)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.cpu()

# ----------------------------
# Face Detection
# ----------------------------
def detect_faces(frame_bgr):
    """Detects faces and returns list of (x,y,w,h) boxes and cropped PIL images."""
    face_boxes = []
    face_pils = []

    if USE_MTCNN and mtcnn is not None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # MTCNN expects PIL image for detection
        # For extracting face crops, it's often easier to let MTCNN do it if `post_process=True`
        # Or detect boxes and then crop from the original frame_bgr
        
        # Option 1: MTCNN provides cropped faces directly (if configured)
        # faces_detected_pil = mtcnn(frame_rgb) # This returns tensors or PIL if post_process=True
        # if faces_detected_pil is not None:
        #     if not isinstance(faces_detected_pil, list): faces_detected_pil = [faces_detected_pil]
        #     for i, face_pil_tensor in enumerate(faces_detected_pil):
        #         # Need bounding boxes too for drawing
        #         # This requires getting boxes from mtcnn.detect separately
        #         # For simplicity, let's get boxes and then crop.

        # Option 2: Get boxes from MTCNN then crop
        img_pil = Image.fromarray(frame_rgb)
        boxes, probs = mtcnn.detect(img_pil) # boxes are [x1, y1, x2, y2]

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                w, h = x2 - x1, y2 - y1
                if w > 0 and h > 0 and x1 >= 0 and y1 >=0 and x2 <= frame_bgr.shape[1] and y2 <= frame_bgr.shape[0]: # Basic check
                    face_boxes.append((x1, y1, w, h))
                    cropped_bgr = frame_bgr[y1:y2, x1:x2]
                    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                    face_pils.append(Image.fromarray(cropped_rgb))
                else:
                    face_pils.append(None) # Add placeholder if box is invalid

    else: # Use Haar Cascade
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)) # Increased minSize
        for (x, y, w, h) in faces:
            face_boxes.append((x, y, w, h))
            cropped_bgr = frame_bgr[y:y+h, x:x+w]
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            face_pils.append(Image.fromarray(cropped_rgb))

    return face_boxes, face_pils


# ----------------------------
# Enrollment (Minor changes for clarity)
# ----------------------------
def enroll_new_person(model):
    person_name = input("Enter the name of the person to enroll: ").strip()
    if not person_name:
        print("Name cannot be empty.")
        return None, None

    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    embeddings_list = []
    images_captured = 0
    source = input("Enroll from (c)am or (f)older? [c/f]: ").lower()

    print("\n--- Enrollment Advice ---")
    print("- Ensure good, consistent lighting.")
    print("- Subject should look directly at the camera or have slight pose variations.")
    print("- Avoid strong shadows, blur, or occlusions (glasses, hair over face).")
    print("- Capture a few slightly different expressions if possible.")
    print("-------------------------\n")


    if source == 'c':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): print("Error: Could not open webcam."); return None, None
        print(f"Look at the camera. Press 'c' to capture, 'q' to quit enrollment.")
        
        while images_captured < NUM_IMAGES_PER_PERSON:
            ret, frame = cap.read()
            if not ret: break
            
            display_frame = frame.copy()
            # For enrollment, we generally want one clear face.
            # MTCNN with select_largest=True might be good here if you adapt detect_faces
            face_boxes_detected, _ = detect_faces(frame) # We only need boxes for display here

            if len(face_boxes_detected) == 1:
                x, y, w, h = face_boxes_detected[0]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Capture {images_captured+1}/{NUM_IMAGES_PER_PERSON}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            # ... (rest of camera enrollment UI is similar) ...
            cv2.imshow("Enrollment - Press 'c' to capture, 'q' to quit", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                # Re-detect to get the PIL image for embedding
                boxes, pils = detect_faces(frame)
                if len(boxes) == 1 and pils[0] is not None:
                    img_path = os.path.join(person_dir, f"{person_name}_{images_captured+1}.jpg")
                    # Save the original detected face region from frame for review
                    x_s, y_s, w_s, h_s = boxes[0]
                    cv2.imwrite(img_path, frame[y_s:y_s+h_s, x_s:x_s+w_s])
                    print(f"Saved raw capture to {img_path}")

                    embedding = get_embedding(pils[0], model)
                    if embedding is not None:
                        embeddings_list.append(embedding)
                        images_captured += 1
                        print(f"Image {images_captured}/{NUM_IMAGES_PER_PERSON} captured and embedding extracted.")
                    else:
                        print("Could not extract embedding from captured face.")
                else:
                    print(f"Could not get a single clear face for capture. Detected: {len(boxes)}")

            elif key == ord('q'): break
        cap.release()
        cv2.destroyAllWindows() # Close only enrollment window

    elif source == 'f':
        # Folder enrollment logic (largely the same, ensure detect_faces is used)
        folder_path = input(f"Enter path to folder containing images for {person_name}: ").strip()
        if not os.path.isdir(folder_path): print("Invalid folder path."); return None, None
        
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:NUM_IMAGES_PER_PERSON] # Limit to NUM_IMAGES
        
        for i, img_path in enumerate(image_files):
            if images_captured >= NUM_IMAGES_PER_PERSON: break
            
            bgr_image = cv2.imread(img_path)
            if bgr_image is None: print(f"Could not read image: {img_path}"); continue

            boxes, pils = detect_faces(bgr_image)
            
            if len(boxes) == 1 and pils[0] is not None:
                # Save a copy to the known_faces_db for consistency (already a cropped face usually)
                db_img_path = os.path.join(person_dir, f"{person_name}_{images_captured+1}.jpg")
                # To save the detected crop:
                x_s, y_s, w_s, h_s = boxes[0]
                cv2.imwrite(db_img_path, bgr_image[y_s:y_s+h_s, x_s:x_s+w_s])
                print(f"Processed and saved detected face to {db_img_path}")

                embedding = get_embedding(pils[0], model)
                if embedding is not None:
                    embeddings_list.append(embedding)
                    images_captured += 1
                    print(f"Image {images_captured}/{NUM_IMAGES_PER_PERSON} from folder processed.")
                else: print(f"Could not extract embedding from {img_path}.")
            else: print(f"Skipping {img_path}: {len(boxes)} faces detected or no valid PIL (expected 1).")


    if not embeddings_list: # Check if any embeddings were generated
        print(f"No valid embeddings captured for {person_name}. Enrollment failed.")
        return None, None
    if len(embeddings_list) < max(1, NUM_IMAGES_PER_PERSON // 2) : # Allow if at least half are good
        print(f"Warning: Only {len(embeddings_list)} embeddings captured for {person_name}. Result might be less robust.")


    person_embedding_avg = torch.mean(torch.cat(embeddings_list, dim=0), dim=0, keepdim=True)
    print(f"Enrollment for {person_name} complete. Average embedding calculated from {len(embeddings_list)} images.")
    return person_name, person_embedding_avg


# ----------------------------
# Load/Save Known Embeddings
# ----------------------------
def load_known_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"Loading known embeddings from {EMBEDDINGS_FILE}...")
        data = torch.load(EMBEDDINGS_FILE, map_location=DEVICE) # Load to current device
        known_embeddings_tensor = data['embeddings_tensor'] # Should be a single tensor
        known_names = data['names']
        print(f"Loaded {len(known_names)} known faces.")
        return known_embeddings_tensor, known_names
    return torch.empty((0, facenet_model.head.in_features if hasattr(facenet_model.head, 'in_features') else 512)).to(DEVICE), [] # Assuming 512 if not found

def save_known_embeddings(embeddings_tensor, names_list):
    if embeddings_tensor.numel() == 0: # Check if tensor is empty
        print("No embeddings to save.")
        return
    print(f"Saving {len(names_list)} known embeddings to {EMBEDDINGS_FILE}...")
    torch.save({'embeddings_tensor': embeddings_tensor.cpu(), 'names': names_list}, EMBEDDINGS_FILE) # Save on CPU
    print("Embeddings saved.")


# ----------------------------
# Real-time Recognition (with Flickering Reduction)
# ----------------------------
def recognize_faces(model, known_embeddings_tensor, known_names, current_threshold):
    global next_face_id, face_recognition_histories
    if known_embeddings_tensor.numel() == 0:
        print("No known faces enrolled. Please enroll faces first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Error: Could not open webcam."); return

    print(f"Starting real-time recognition with threshold: {current_threshold}. Press 'q' to quit.")
    
    # Simple face tracking (assign an ID to a face if it's close to a previous one)
    # For simplicity, we'll just use the order of detection in a frame as a temporary ID for history lookup.
    # A more robust solution would use an actual object tracker.
    
    active_face_histories = {} # Store history for faces seen in *this* frame loop

    frame_count = 0
    fps = 0
    start_time = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret: break

        frame_display = frame_bgr.copy()
        detected_face_boxes, detected_face_pils = detect_faces(frame_bgr)
        
        current_frame_face_data = [] # Store (box, name, score) for this frame

        for i, (box, pil_image) in enumerate(zip(detected_face_boxes, detected_face_pils)):
            x, y, w, h = box
            
            final_name = "Processing..."
            final_color = (255, 165, 0) # Orange for processing
            min_dist_val = float('inf')

            if pil_image is not None:
                current_embedding = get_embedding(pil_image, model)
                if current_embedding is not None:
                    # Distances to all known embeddings (tensorized)
                    # Ensure current_embedding is also on DEVICE (get_embedding should handle it)
                    distances = torch.cdist(current_embedding.to(DEVICE), known_embeddings_tensor.to(DEVICE)).squeeze()
                    
                    if distances.numel() > 0:
                        min_dist, min_idx = torch.min(distances, dim=0)
                        min_dist_val = min_dist.item()
                        
                        predicted_name = "Unknown"
                        if min_dist_val < current_threshold:
                            predicted_name = known_names[min_idx.item()]
                    else: # No known embeddings
                        predicted_name = "Unknown"
                        min_dist_val = float('inf')


                    # --- Flickering Reduction Logic ---
                    # Use a simple ID based on detection order for this example
                    # A real tracker would provide more stable IDs
                    face_id_in_frame = i 

                    if face_id_in_frame not in active_face_histories:
                        active_face_histories[face_id_in_frame] = {
                            "name_history": deque(maxlen=RECOGNITION_HISTORY_SIZE),
                            "score_history": deque(maxlen=RECOGNITION_HISTORY_SIZE),
                            "current_stable_name": "Unknown",
                            "stability_count": 0
                        }
                    
                    history = active_face_histories[face_id_in_frame]
                    history["name_history"].append(predicted_name)
                    history["score_history"].append(min_dist_val)

                    # Check for stability
                    if len(history["name_history"]) >= CONFIDENCE_STABILITY_THRESHOLD:
                        # Consider the last N predictions
                        last_n_names = list(history["name_history"])[-CONFIDENCE_STABILITY_THRESHOLD:]
                        if all(name == last_n_names[0] for name in last_n_names):
                            if history["current_stable_name"] != last_n_names[0]:
                                history["current_stable_name"] = last_n_names[0]
                                history["stability_count"] = CONFIDENCE_STABILITY_THRESHOLD
                            else:
                                history["stability_count"] = min(history["stability_count"] + 1, RECOGNITION_HISTORY_SIZE)
                        # else if not stable, current_stable_name might remain or be reset based on policy
                        # For now, if it becomes unstable, it might revert to the most recent raw prediction or "Processing"

                    final_name = history["current_stable_name"]
                    if final_name == "Unknown":
                        final_color = (0, 0, 255) # Red
                    elif final_name != "Processing...":
                        final_color = (0, 255, 0) # Green
                    
                    # If you want to display the raw score of the current_stable_name:
                    # Find its most recent score in history if it matches current_stable_name
                    display_score = min_dist_val # Default to current frame's score
                    try:
                        # Get score corresponding to the stable name, if available recently
                        indices = [j for j, name in enumerate(history["name_history"]) if name == final_name]
                        if indices:
                            display_score = history["score_history"][indices[-1]]
                    except:
                        pass # Keep default score

                    text = f"{final_name} ({display_score:.2f})"
                    current_frame_face_data.append(((x,y,w,h), text, final_color))
                else: # No embedding extracted
                    current_frame_face_data.append(((x,y,w,h), "No Emb", (0,100,255)))
            else: # No PIL image (bad detection)
                 current_frame_face_data.append(((x,y,w,h), "Bad Det", (0,0,100)))
        
        # Clean up histories for faces not detected in this frame (simplified)
        # In a real system, you'd have a tracker that tells you when a track is lost.
        current_ids_present = set(range(len(detected_face_boxes)))
        ids_to_remove = [fid for fid in active_face_histories if fid not in current_ids_present]
        for fid in ids_to_remove:
            del active_face_histories[fid]


        # Draw all faces at the end
        for box_coords, text_to_display, color_to_use in current_frame_face_data:
            x_d, y_d, w_d, h_d = box_coords
            cv2.rectangle(frame_display, (x_d, y_d), (x_d + w_d, y_d + h_d), color_to_use, 2)
            cv2.putText(frame_display, text_to_display, (x_d, y_d - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_to_use, 2)


        # FPS Calculation
        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            fps = frame_count / (end_time - start_time) if (end_time - start_time) > 0 else 0
            start_time = time.time()
            frame_count = 0
        cv2.putText(frame_display, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame_display, f"Threshold: {current_threshold:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow('Real-time Face Recognition', frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('+') or key == ord('='): current_threshold += 0.05; print(f"Threshold: {current_threshold:.2f}")
        elif key == ord('-'): current_threshold -= 0.05; print(f"Threshold: {current_threshold:.2f}")


    cap.release()
    cv2.destroyAllWindows()
    return current_threshold # Return the potentially adjusted threshold

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    if not USE_MTCNN and not os.path.exists(HAARCASCADE_PATH):
        print(f"Error: Haar Cascade XML file '{HAARCASCADE_PATH}' not found.")
        exit()

    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    
    facenet_model = load_trained_model(MODEL_PATH)
    
    # Try to get embedding dimension from model
    try:
        # If model.head was a Linear layer before Identity, its in_features would be embedding dim
        # For FasterViT, this might be a specific value depending on the variant (e.g., 512, 768)
        # Let's assume 512 if we can't determine it. Your 'faster_vit_0_224' likely outputs 512 features before head.
        EMBEDDING_DIM = 512
        # Example: if model.head was nn.Linear(in_features=X, out_features=Y)
        # if isinstance(facenet_model.head, torch.nn.Linear): EMBEDDING_DIM = facenet_model.head.in_features
        # Since it's Identity(), we rely on the known output dim of the backbone.
        print(f"Assuming embedding dimension: {EMBEDDING_DIM}")
    except Exception as e:
        print(f"Could not reliably determine embedding dim, defaulting to 512. Error: {e}")
        EMBEDDING_DIM = 512


    # Load known embeddings as a single tensor
    loaded_embeddings, loaded_names = load_known_embeddings()
    if loaded_embeddings.numel() > 0 and loaded_embeddings.shape[1] != EMBEDDING_DIM:
        print(f"Warning: Loaded embeddings dim ({loaded_embeddings.shape[1]}) mismatch assumed dim ({EMBEDDING_DIM}). This may cause errors.")
        # Handle mismatch: either re-enroll or ensure consistency. For now, proceed with caution.
    
    # Convert list of tensors to a single tensor for efficient cdist
    known_face_embeddings_tensor = loaded_embeddings
    known_face_names = loaded_names

    # Initial recognition threshold (TUNE THIS!)
    recognition_threshold = 0.85 # Start with a value and adjust based on testing

    while True:
        action = input(f"\nThreshold: {recognition_threshold:.2f} | Choose: (e)nroll, (r)ecognize, (s)ave, (t)une threshold, (q)uit: ").lower()
        if action == 'e':
            name, avg_emb = enroll_new_person(facenet_model)
            if name and avg_emb is not None:
                if name in known_face_names:
                    idx = known_face_names.index(name)
                    known_face_embeddings_tensor[idx] = avg_emb.squeeze() # Squeeze to match tensor dims
                    print(f"Updated embedding for: {name}")
                else:
                    if known_face_embeddings_tensor.numel() == 0: # First enrollment
                         known_face_embeddings_tensor = avg_emb.to(DEVICE) # Keep on device
                    else:
                         known_face_embeddings_tensor = torch.cat((known_face_embeddings_tensor, avg_emb.to(DEVICE)), dim=0)
                    known_face_names.append(name)
                    print(f"Enrolled new person: {name}")
        elif action == 'r':
            if known_face_embeddings_tensor.numel() == 0:
                print("No faces enrolled yet.")
            else:
                print("During recognition: Press '+' to increase threshold, '-' to decrease.")
                recognition_threshold = recognize_faces(facenet_model, known_face_embeddings_tensor, known_face_names, recognition_threshold)
        elif action == 's':
            save_known_embeddings(known_face_embeddings_tensor, known_face_names)
        elif action == 't':
            try:
                new_thresh = float(input(f"Enter new threshold (current: {recognition_threshold:.2f}): "))
                recognition_threshold = new_thresh
                print(f"Threshold set to: {recognition_threshold:.2f}")
            except ValueError:
                print("Invalid input for threshold.")
        elif action == 'q':
            print("Exiting.")
            break
        else:
            print("Invalid action.")