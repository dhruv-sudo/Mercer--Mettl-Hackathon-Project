import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

def get_face_score(image_path):
    # 1. Setup the new MediaPipe Face Landmarker Tasks API
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        # 2. Load and process image using standard OpenCV and MediaPipe Images
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return 0.0
            
        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # 3. Detect Landmarks
        detection_result = landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return 0.0
            
        landmarks = detection_result.face_landmarks[0]
        
        # Helper function to convert normalized positions to actual pixel coords
        def get_pixel_coords(idx):
            pt = landmarks[idx]
            return np.array([int(pt.x * w), int(pt.y * h)])
            
        # Get your points (using the same landmark indexes)
        left_eye = get_pixel_coords(33)
        right_eye = get_pixel_coords(263)
        left_mouth = get_pixel_coords(61)
        right_mouth = get_pixel_coords(291)
        
        # Calculate real pixel distances
        eye_dist = np.linalg.norm(left_eye - right_eye)
        mouth_dist = np.linalg.norm(left_mouth - right_mouth)
        
        # Standardize score logic safely
        score = mouth_dist / eye_dist if eye_dist > 0 else 0.0
        return min(max(score, 0.0), 1.0)
