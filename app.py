import io
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
from passlib.context import CryptContext
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

# Image dimensions
IMG_WIDTH = 30
IMG_HEIGHT = 30

# Traffic sign categories
SIGN_CATEGORIES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve to the left",
    20: "Dangerous curve to the right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead",
    34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right",
    37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

# Initialize FastAPI
app = FastAPI(
    title="Traffic Sign Classification API",
    description="Production-grade API with authentication and database logging",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Load Model
MODEL = None

@app.on_event("startup")
async def startup_event():
    global MODEL
    try:
        MODEL = tf.keras.models.load_model("traffic_model.h5")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# Pydantic Models
class UserSignup(BaseModel):
    email: EmailStr
    password: str
    username: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class PredictionResponse(BaseModel):
    prediction_id: str
    class_id: int
    sign_name: str
    confidence: float
    top_3_predictions: List[dict]
    timestamp: str

class PredictionHistory(BaseModel):
    predictions: List[dict]
    total: int

# Helper Functions
def preprocess_image(image: Image.Image):
    """Resizes and prepares image for the model."""
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return {"user_id": user_id, "email": email}
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# Routes
@app.get("/")
def home():
    return {
        "message": "Traffic Sign Classification API v2.0",
        "status": "online",
        "model_loaded": MODEL is not None,
        "database_connected": supabase is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "signup": "/auth/signup",
            "login": "/auth/login",
            "predict": "/predict"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "database_connected": supabase is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/auth/signup", response_model=Token)
async def signup(user: UserSignup):
    """Create a new user account"""
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not configured"
        )
    
    try:
        # Sign up with Supabase Auth
        auth_response = supabase.auth.sign_up({
            "email": user.email,
            "password": user.password
        })
        
        if not auth_response.user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not create user"
            )
        
        # Store additional user info
        supabase.table("users").insert({
            "id": auth_response.user.id,
            "email": user.email,
            "username": user.username,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        # Create JWT token
        access_token = create_access_token(
            data={"sub": auth_response.user.id, "email": user.email},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": auth_response.user.id,
                "email": user.email,
                "username": user.username
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Signup failed: {str(e)}"
        )

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    """Login to get access token"""
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not configured"
        )
    
    try:
        # Sign in with Supabase Auth
        auth_response = supabase.auth.sign_in_with_password({
            "email": user.email,
            "password": user.password
        })
        
        if not auth_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Get user info
        user_info = supabase.table("users").select("*").eq("id", auth_response.user.id).execute()
        username = user_info.data[0]["username"] if user_info.data else user.email
        
        # Create JWT token
        access_token = create_access_token(
            data={"sub": auth_response.user.id, "email": user.email},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": auth_response.user.id,
                "email": user.email,
                "username": username
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Predict traffic sign with authentication"""
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Accept common image extensions including .ppm
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.gif', '.tiff']
    file_ext = file.filename.lower().split('.')[-1] if file.filename else ''
    
    if not (file.content_type and file.content_type.startswith("image/")) and f'.{file_ext}' not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File must be an image. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = MODEL.predict(processed_image, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                "class_id": int(idx),
                "sign_name": SIGN_CATEGORIES.get(int(idx), f"Category {idx}"),
                "confidence": float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        # Save to database
        prediction_record = None
        if supabase:
            try:
                result = supabase.table("predictions").insert({
                    "user_id": current_user["user_id"],
                    "class_id": predicted_class,
                    "sign_name": SIGN_CATEGORIES.get(predicted_class, "Unknown"),
                    "confidence": confidence,
                    "filename": file.filename,
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                prediction_record = result.data[0]["id"] if result.data else "not_saved"
            except Exception as e:
                print(f"Warning: Could not save to database: {e}")
                prediction_record = "not_saved"
        
        return {
            "prediction_id": prediction_record or "no_database",
            "class_id": predicted_class,
            "sign_name": SIGN_CATEGORIES.get(predicted_class, "Unknown"),
            "confidence": confidence,
            "top_3_predictions": top_3_predictions,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/predictions/history", response_model=PredictionHistory)
async def get_prediction_history(
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get user's prediction history"""
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not configured"
        )
    
    try:
        result = supabase.table("predictions")\
            .select("*")\
            .eq("user_id", current_user["user_id"])\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        return {
            "predictions": result.data,
            "total": len(result.data)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not fetch history: {str(e)}"
        )

@app.get("/categories")
async def get_categories():
    """Get all traffic sign categories"""
    return {
        "categories": SIGN_CATEGORIES,
        "total": len(SIGN_CATEGORIES)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)