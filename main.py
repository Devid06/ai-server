import face_recognition
import numpy as np
import requests
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# --- Pydantic Models for Request/Response ---

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

# --- FastAPI App Initialization ---

app = FastAPI(
    title="Spark Face Verification API",
    description="A custom API for facial recognition and embedding generation."
)

# --- Helper Function ---

def load_image_from_url(url: str):
    """
    Downloads an image from a URL and loads it into a format
    face_recognition can understand.
    """
    try:
        response = requests.get(url)
        response.raise_for_status() # Raises an HTTPError for bad responses
        
        # Load image from in-memory bytes
        image = face_recognition.load_image_file(io.BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download image from URL: {url}")
    except Exception as e:
        # This could be an unsupported image format
        print(f"Error loading image: {e}")
        raise HTTPException(status_code=400, detail=f"Could not load image. It may be corrupt or an unsupported format.")


# --- API Endpoints ---

@app.post("/verify-face", response_model=FaceVerifyResponse)
async def verify_face(request: FaceVerifyRequest):
    """
    Endpoint 1: 1:1 Face Match
    Compares a live selfie to a user's profile picture.
    """
    try:
        # 1. Load both images from their URLs
        live_selfie_img = load_image_from_url(request.liveSelfieUrl)
        profile_pic_img = load_image_from_url(request.profilePicUrl)

        # 2. Get face encodings (the facial features)
        # We only take the first face found in each image
        live_encodings = face_recognition.face_encodings(live_selfie_img)
        profile_encodings = face_recognition.face_encodings(profile_pic_img)

        # 3. Handle cases where no face is found
        if not live_encodings:
            raise HTTPException(status_code=400, detail="No face detected in the live selfie.")
        if not profile_encodings:
            raise HTTPException(status_code=400, detail="No face detected in the profile picture.")

        live_encoding = live_encodings[0]
        profile_encoding = profile_encodings[0]

        # 4. Compare the faces
        # compare_faces returns a list of [True] or [False]
        is_same = face_recognition.compare_faces([profile_encoding], live_encoding)[0]
        
        # 5. Calculate similarity (1.0 = identical, 0.0 = very different)
        # face_distance returns a "distance" (lower is better).
        # We subtract from 1.0 to get "similarity" (higher is better).
        face_distance = face_recognition.face_distance([profile_encoding], live_encoding)[0]
        similarity = 1.0 - face_distance

        return FaceVerifyResponse(
            isSamePerson=bool(is_same), # Convert numpy.bool_ to standard bool
            similarity=similarity
        )

    except HTTPException as e:
        # Re-raise HTTP exceptions from the helper
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during face verification: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during face comparison.")


@app.post("/generate-embedding", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    """
    Endpoint 2: 1:N Embedding Generation
    Generates a 128-dimension vector for a given image.
    """
    try:
        # 1. Load the image
        image = load_image_from_url(request.imageUrl)

        # 2. Get face encoding
        encodings = face_recognition.face_encodings(image)

        # 3. Handle no face found
        if not encodings:
            raise HTTPException(status_code=400, detail="No face detected in the image.")

        # 4. Get the first encoding and convert from numpy array to a standard list
        embedding = encodings[0].tolist()

        return EmbeddingResponse(embedding=embedding)

    except HTTPException as e:
        # Re-raise HTTP exceptions from the helper
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during embedding generation: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during embedding generation.")


# --- Run the Server ---

if __name__ == "__main__":
    import uvicorn
    # This runs the server. Uvicorn is a high-speed server for FastAPI.
    # host="0.0.0.0" makes it accessible on your network, not just localhost.
    uvicorn.run(app, host="0.0.0.0", port=8000)