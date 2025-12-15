import face_recognition
import numpy as np
import requests
import io
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# --- Pydantic Models ---
class FaceVerifyRequest(BaseModel):
    liveSelfieUrl: str
    profilePicUrl: str

class FaceVerifyResponse(BaseModel):
    isSamePerson: bool
    similarity: float

class EmbeddingRequest(BaseModel):
    imageUrl: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

# --- FastAPI App ---
app = FastAPI(
    title="Spark Face Verification API",
    description="A custom API for facial recognition and embedding generation."
)

# --- Helper Function ---
def load_image_from_url(url: str):
    try:
        response = requests.get(url, timeout=10) # Added timeout
        response.raise_for_status()
        
        # Load image
        image = face_recognition.load_image_file(io.BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        # Return the EXACT error from the network request
        raise HTTPException(status_code=400, detail=f"Image download failed: {str(e)}")
    except Exception as e:
        print(f"Error loading image data: {e}")
        # Return the EXACT error from face_recognition
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

# --- Endpoints ---

@app.post("/verify-face", response_model=FaceVerifyResponse)
async def verify_face(request: FaceVerifyRequest):
    try:
        # 1. Load images
        img1 = load_image_from_url(request.liveSelfieUrl)
        img2 = load_image_from_url(request.profilePicUrl)

        # 2. Get encodings
        encodings1 = face_recognition.face_encodings(img1)
        encodings2 = face_recognition.face_encodings(img2)

        if not encodings1 or not encodings2:
            raise HTTPException(status_code=400, detail="No face detected in one or both images.")

        # 3. Compare
        encoding1 = encodings1[0]
        encoding2 = encodings2[0]

        # Calculate distance (lower is better)
        face_distance = face_recognition.face_distance([encoding1], encoding2)[0]
        
        # Convert distance to similarity score (0 to 1)
        # 0.6 is the typical threshold for dlib. 
        similarity = max(0.0, 1.0 - face_distance)
        
        # We consider it a match if distance is less than 0.6 (so similarity > 0.4)
        is_same = face_distance < 0.6

        return FaceVerifyResponse(isSamePerson=is_same, similarity=similarity)

    except HTTPException as e:
        raise e
    except Exception as e:
        # LOG THE REAL ERROR
        error_msg = f"Verify Crash: {str(e)}\n{traceback.format_exc()}"
        print(error_msg) 
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


@app.post("/generate-embedding", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    try:
        image = load_image_from_url(request.imageUrl)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            raise HTTPException(status_code=400, detail="No face detected in the image.")

        embedding = encodings[0].tolist()
        return EmbeddingResponse(embedding=embedding)

    except HTTPException as e:
        raise e
    except Exception as e:
        # LOG THE REAL ERROR
        error_msg = f"Embedding Crash: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
