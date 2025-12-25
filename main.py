import face_recognition
import numpy as np
import requests
import cv2  # OpenCV for quality checks
import io
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# --- CONFIGURATION ---
MIN_BRIGHTNESS = 40   # 0-255 (Reject dark photos)
MIN_SHARPNESS = 60    # Higher = Sharper (Reject blurry photos)

# --- Pydantic Models ---
class FaceVerifyRequest(BaseModel):
    liveSelfieUrl: str
    profilePicUrl: str

class FaceVerifyResponse(BaseModel):
    isSamePerson: bool
    similarity: float
    distance: float       # NEW: Return raw distance for debugging
    warning: str = None   # NEW: Warn if quality is borderline

class EmbeddingRequest(BaseModel):
    imageUrl: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

# --- FastAPI App ---
app = FastAPI(
    title="Spark AI Server v2.0",
    description="High-Accuracy Face Verification with Quality Control."
)

# --- Helper Functions ---
def load_image_from_url(url: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Convert to numpy array for OpenCV
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("Could not decode image.")
            
        # Convert BGR (OpenCV) to RGB (face_recognition)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, img_bgr
        
    except Exception as e:
        print(f"Download Error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

def check_quality(img_bgr):
    # 1. Check Brightness
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    
    # 2. Check Blur (Laplacian Variance)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    print(f"[Quality Check] Brightness: {brightness:.1f}, Sharpness: {sharpness:.1f}")
    
    if brightness < MIN_BRIGHTNESS:
        raise HTTPException(status_code=400, detail="Photo is too dark. Please find better lighting.")
    
    if sharpness < MIN_SHARPNESS:
        raise HTTPException(status_code=400, detail="Photo is too blurry. Please hold steady.")
        
    return True

# --- Endpoints ---

@app.post("/verify-face", response_model=FaceVerifyResponse)
async def verify_face(request: FaceVerifyRequest):
    try:
        # 1. Load & Check Quality
        print(f"Processing 1:1 Verification...")
        img1_rgb, img1_bgr = load_image_from_url(request.liveSelfieUrl)
        img2_rgb, img2_bgr = load_image_from_url(request.profilePicUrl)
        
        # Validate Quality (Selfie is most critical)
        check_quality(img1_bgr)
        # We can be lenient with profile pic, or check it too:
        # check_quality(img2_bgr)

        # 2. Get Encodings (Upsample 1x for better small-face detection)
        encodings1 = face_recognition.face_encodings(img1_rgb, num_jitters=1)
        encodings2 = face_recognition.face_encodings(img2_rgb, num_jitters=1)

        if not encodings1 or not encodings2:
            raise HTTPException(status_code=400, detail="No face detected. Please face the camera directly.")

        # 3. Compare
        encoding1 = encodings1[0]
        encoding2 = encodings2[0]

        # Distance: 0.0 (Same) -> 1.0 (Different)
        face_distance = face_recognition.face_distance([encoding1], encoding2)[0]
        
        # Convert to Similarity: 1.0 (Same) -> 0.0 (Different)
        similarity = max(0.0, 1.0 - face_distance)
        
        # Strict Threshold for 1:1
        is_same = face_distance < 0.55  # Slightly stricter than default 0.6

        print(f"[Verify] Distance: {face_distance:.4f} | Similarity: {similarity:.4f} | Match: {is_same}")

        return FaceVerifyResponse(
            isSamePerson=is_same, 
            similarity=similarity,
            distance=face_distance
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        error_msg = f"Verify Crash: {str(e)}\n{traceback.format_exc()}"
        print(error_msg) 
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


@app.post("/generate-embedding", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    try:
        # 1. Load & Check Quality
        print(f"Generating Embedding...")
        img_rgb, img_bgr = load_image_from_url(request.imageUrl)
        
        check_quality(img_bgr)

        # 2. Generate Encoding
        encodings = face_recognition.face_encodings(img_rgb, num_jitters=1)

        if not encodings:
            raise HTTPException(status_code=400, detail="No face detected in the image.")

        # 3. Return Vector
        embedding = encodings[0].tolist()
        print(f"[Embedding] Generated vector of size {len(embedding)}")
        
        return EmbeddingResponse(embedding=embedding)

    except HTTPException as e:
        raise e
    except Exception as e:
        error_msg = f"Embedding Crash: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
