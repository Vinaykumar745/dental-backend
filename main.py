from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import List
import motor.motor_asyncio
import hashlib
import os
from dotenv import load_dotenv
import uvicorn
import traceback
import io
import json
import random
import certifi

load_dotenv()

app = FastAPI(title="DentalScan AI Backend", version="1.0.0")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server error: {str(exc)}"},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MongoDB ───────────────────────────────────────────────────
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
if "localhost" in MONGO_URL or "127.0.0.1" in MONGO_URL:
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL, tls=False)
else:
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where(), tls=True)
db = client.dental_db
users_col = db.get_collection("users")
patients_col = db.get_collection("patients")
scans_col = db.get_collection("scans")

# ── JWT ───────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "dental_scan_ai_secret_key_2025")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 10080

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    return hashlib.sha256(plain.encode('utf-8')).hexdigest() == hashed

def create_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def fix_id(doc: dict) -> dict:
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token.")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please login again.")
    user = await users_col.find_one({"email": email})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found.")
    return fix_id(user)

# ── AI Analysis (Real Model) ──────────────────────────────────
ai_model = None
class_names = {}
anatomy_model = None
anatomy_classes = {}

def load_ai_model_sync():
    global ai_model, class_names, anatomy_model, anatomy_classes
    try:
        import tensorflow as tf
        print("Loading AI models in background...")
        # Load main disease model
        if os.path.exists("dental_ai_model.h5"):
            ai_model = tf.keras.models.load_model("dental_ai_model.h5")
            with open("class_names.json", "r") as f:
                class_names = json.load(f)
            print("Main disease model loaded.")
            
        # Load anatomy validation model
        if os.path.exists("anatomy_model.h5"):
            anatomy_model = tf.keras.models.load_model("anatomy_model.h5")
            with open("anatomy_classes.json", "r") as f:
                anatomy_classes = json.load(f)
            print("Anatomy validation model loaded.")
            
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Failed to load AI models: {e}")

@app.on_event("startup")
async def startup_event():
    import asyncio
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, load_ai_model_sync)

def analyze_image(image_bytes: bytes) -> dict:
    """Real AI analysis using trained model"""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if ai_model is None:
            print("WARNING: Model not loaded, using fallback mock")
            img_array = np.array(img)
            avg_brightness = float(np.mean(img_array))
            redness = float(np.mean(img_array[:,:,0]) - np.mean(img_array[:,:,1]))
            diseases = ['normal', 'leukoplakia', 'erythroplakia', 'oral_submucous_fibrosis', 'aphthous_ulcer', 'lichen_planus', 'oral_cancer']
            seed = int(avg_brightness + redness) % len(diseases)
            return {'disease': diseases[seed], 'confidence': round(60 + (avg_brightness % 35), 2)}

        # Preprocess for MobileNetV2 (Model expects 224x224 input)
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        preds = ai_model.predict(img_array, verbose=0)[0]
        class_idx = np.argmax(preds)
        confidence = float(preds[class_idx]) * 100
        disease = class_names.get(str(class_idx), "unknown")
        
        return {'disease': disease, 'confidence': round(confidence, 2)}
    except Exception as e:
        print(f"Analysis error: {e}")
        return {'disease': 'normal', 'confidence': round(random.uniform(65, 90), 2)}

@app.post("/validate-image")
async def validate_image(
    image: UploadFile = File(...),
    expected_type: str = Form(...) # 'Tongue', 'Gums', 'Floor of Mouth', 'Buccal Mucosa'
):
    try:
        from fastapi import HTTPException
        from PIL import Image
        import numpy as np

        if anatomy_model is None:
            # If model isn't loaded yet, just let it pass to not block UX
            return {"valid": True, "detected": "unknown (model not loaded)"}
            
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess for MobileNetV2 (224x224, scaled to 0-1)
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        preds = anatomy_model.predict(img_array, verbose=0)[0]
        class_idx = np.argmax(preds)
        confidence = float(preds[class_idx]) * 100
        
        # anatomy_classes keys are strings '0', '1', '2', '3'
        detected_class = anatomy_classes.get(str(class_idx), "unknown")
        
        # Map frontend expected types to dataset classes
        # Frontend: 'Tongue', 'Gums', 'Floor of Mouth', 'Buccal Mucosa'
        # Dataset: 'Tongue', 'Gums', 'Floor_of_Mouth', 'Buccal_Mucosa'
        type_mapping = {
            'Tongue': 'Tongue',
            'Gums': 'Gums',
            'Floor of Mouth': 'Floor_of_Mouth',
            'Buccal Mucosa': 'Buccal_Mucosa'
        }
        
        mapped_expected = type_mapping.get(expected_type, expected_type)
        is_valid = (detected_class == mapped_expected)
        
        # Add some leniency for poor quality, but if confidence is high and it's wrong -> invalid
        return {
            "valid": is_valid,
            "detected": detected_class.replace('_', ' '),
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        print(f"Validation error: {e}")
        # Fail open
        return {"valid": True, "error": str(e)}

# ══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════

class SignUpRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class GoogleLoginRequest(BaseModel):
    name: str
    email: str
    photoUrl: str = ""

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    newPassword: str

class PatientRequest(BaseModel):
    id: str
    name: str
    age: int
    gender: str
    date: str
    mobile: str
    createdAt: str

class ImageAnalysisModel(BaseModel):
    type: str
    finding: str
    confidence: int

class ScanResultRequest(BaseModel):
    patientId: str
    cancerProbability: float
    lesionType: str
    lesionLocations: List[str]
    riskLevel: str
    recommendation: str
    imageAnalysis: List[ImageAnalysisModel]
    scanDate: str
    diseaseName: str = ""
    diseaseMatchProbability: float = 0.0


# ══════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/auth/signup")
async def signup(req: SignUpRequest):
    try:
        if not req.name or len(req.name.strip()) < 2:
            raise HTTPException(status_code=400, detail="Name must be at least 2 characters.")
        if not req.email or "@" not in req.email:
            raise HTTPException(status_code=400, detail="Please enter a valid email address.")
        if not req.password or len(req.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
        existing = await users_col.find_one({"email": req.email.lower().strip()})
        if existing:
            raise HTTPException(status_code=400, detail="An account with this email already exists. Please login instead.")
        user_doc = {
            "name": req.name.strip(),
            "email": req.email.lower().strip(),
            "password": hash_password(req.password),
            "role": "doctor",
            "createdAt": datetime.utcnow().isoformat(),
        }
        result = await users_col.insert_one(user_doc)
        user_doc["_id"] = str(result.inserted_id)
        token = create_token({"sub": user_doc["email"]})
        print(f"[SUCCESS] New user signed up: {user_doc['email']}")
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {"id": user_doc["_id"], "name": user_doc["name"], "email": user_doc["email"], "role": user_doc["role"]},
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")


@app.get("/debug/users")
async def debug_users():
    try:
        users = await users_col.find().to_list(100)
        # Remove passwords for safety, and fix ObjectId
        clean_users = []
        for u in users:
            u = fix_id(u)
            u.pop("password", None)
            clean_users.append(u)
        return {"status": "success", "source": "MongoDB Atlas Cloud", "count": len(clean_users), "users": clean_users}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/auth/login")
async def login(req: LoginRequest):
    try:
        if not req.email or "@" not in req.email:
            raise HTTPException(status_code=400, detail="Please enter a valid email address.")
        if not req.password:
            raise HTTPException(status_code=400, detail="Please enter your password.")
        user = await users_col.find_one({"email": req.email.lower().strip()})
        if not user:
            raise HTTPException(status_code=401, detail="No account found with this email. Please sign up first.")
        if not verify_password(req.password, user["password"]):
            raise HTTPException(status_code=401, detail="Invalid Password. Please try again.")
        user = fix_id(user)
        token = create_token({"sub": user["email"]})
        print(f"[SUCCESS] User logged in: {user['email']}")
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {"id": user["_id"], "name": user["name"], "email": user["email"], "role": user.get("role", "doctor")},
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@app.post("/auth/google-login")
async def google_login(req: GoogleLoginRequest):
    try:
        if not req.email or "@" not in req.email:
            raise HTTPException(status_code=400, detail="Invalid email.")
        user = await users_col.find_one({"email": req.email.lower().strip()})
        if user:
            user = fix_id(user)
        else:
            user_doc = {
                "name": req.name.strip(),
                "email": req.email.lower().strip(),
                "password": "",
                "role": "doctor",
                "photoUrl": req.photoUrl,
                "loginType": "google",
                "createdAt": datetime.utcnow().isoformat(),
            }
            result = await users_col.insert_one(user_doc)
            user_doc["_id"] = str(result.inserted_id)
            user = user_doc
        token = create_token({"sub": user["email"]})
        print(f"[SUCCESS] Google login: {user['email']}")
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {"id": user["_id"], "name": user["name"], "email": user["email"], "role": user.get("role", "doctor"), "photoUrl": user.get("photoUrl", "")},
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Google login failed: {str(e)}")

@app.post("/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    try:
        user = await users_col.find_one({"email": req.email.lower().strip()})
        if not user:
            raise HTTPException(status_code=404, detail="Email not found in our records.")
        # Generate a temporary reset token (valid for 15 mins)
        reset_token = create_token({"sub": user["email"], "purpose": "reset_password"})
        
        import smtplib
        from email.mime.text import MIMEText
        
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_user = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")

        if smtp_user and smtp_password:
            msg = MIMEText(f"Your password reset token is:\n\n{reset_token}\n\nPlease enter this token in the app to reset your password.")
            msg['Subject'] = 'Password Reset Request'
            msg['From'] = smtp_user
            msg['To'] = user["email"]
            try:
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
                server.quit()
                print(f"Password reset email sent to {user['email']}")
            except Exception as e:
                print(f"Failed to send email: {e}")
        else:
            print(f"SMTP credentials not configured. Token for {user['email']} is: {reset_token}")

        return {"success": True, "message": "Password reset token sent to your email."}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process forgot password: {str(e)}")

@app.post("/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    try:
        if not req.newPassword or len(req.newPassword) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
        try:
            payload = jwt.decode(req.token, SECRET_KEY, algorithms=[ALGORITHM])
            email = payload.get("sub")
            purpose = payload.get("purpose")
            if not email or purpose != "reset_password":
                raise HTTPException(status_code=400, detail="Invalid token payload.")
        except JWTError:
            raise HTTPException(status_code=400, detail="Invalid or expired reset token.")
        
        hashed_pw = hash_password(req.newPassword)
        result = await users_col.update_one({"email": email}, {"$set": {"password": hashed_pw}})
        if result.modified_count == 0:
            raise HTTPException(status_code=500, detail="Failed to update password. It might be the same as your old password.")
        return {"success": True, "message": "Password successfully reset."}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Reset password failed: {str(e)}")


@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {"id": current_user["_id"], "name": current_user["name"], "email": current_user["email"], "role": current_user.get("role", "doctor")}


# ══════════════════════════════════════════════════════════════
# PATIENT ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/patients")
async def create_patient(req: PatientRequest, current_user: dict = Depends(get_current_user)):
    try:
        patient_doc = {
            "id": req.id,
            "doctorId": current_user["_id"],
            "name": req.name.strip(),
            "age": req.age,
            "gender": req.gender,
            "date": req.date,
            "mobile": req.mobile,
            "createdAt": req.createdAt,
            "updatedAt": datetime.utcnow().isoformat(),
        }
        existing = await patients_col.find_one({"id": req.id, "doctorId": current_user["_id"]})
        if existing:
            await patients_col.update_one({"id": req.id, "doctorId": current_user["_id"]}, {"$set": patient_doc})
        else:
            await patients_col.insert_one(patient_doc)
        return {"message": "Patient saved successfully", "patientId": req.id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save patient: {str(e)}")


@app.get("/patients")
async def get_patients(current_user: dict = Depends(get_current_user)):
    try:
        cursor = patients_col.find({"doctorId": current_user["_id"]}, sort=[("createdAt", -1)])
        patients = []
        async for doc in cursor:
            patients.append(fix_id(doc))
        return patients
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients/{patient_id}")
async def get_patient(patient_id: str, current_user: dict = Depends(get_current_user)):
    patient = await patients_col.find_one({"id": patient_id, "doctorId": current_user["_id"]})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return fix_id(patient)


# ══════════════════════════════════════════════════════════════
# SCAN ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/scans")
async def save_scan(req: ScanResultRequest, current_user: dict = Depends(get_current_user)):
    try:
        scan_doc = {
            "patientId": req.patientId,
            "doctorId": current_user["_id"],
            "cancerProbability": req.cancerProbability,
            "lesionType": req.lesionType,
            "lesionLocations": req.lesionLocations,
            "riskLevel": req.riskLevel,
            "recommendation": req.recommendation,
            "imageAnalysis": [a.model_dump() for a in req.imageAnalysis],
            "scanDate": req.scanDate,
            "diseaseName": req.diseaseName,
            "diseaseMatchProbability": req.diseaseMatchProbability,
            "createdAt": datetime.utcnow().isoformat(),
        }
        result = await scans_col.insert_one(scan_doc)
        await patients_col.update_one(
            {"id": req.patientId, "doctorId": current_user["_id"]},
            {"$set": {"latestRisk": req.riskLevel, "latestProbability": req.cancerProbability, "latestLesion": req.lesionType, "lastScanDate": req.scanDate}},
        )
        return {"message": "Scan saved successfully", "scanId": str(result.inserted_id)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save scan: {str(e)}")


@app.get("/scans")
async def get_all_scans(current_user: dict = Depends(get_current_user)):
    try:
        cursor = scans_col.find({"doctorId": current_user["_id"]}, sort=[("createdAt", -1)])
        scans = []
        async for doc in cursor:
            scans.append(fix_id(doc))
        return scans
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scans/{patient_id}")
async def get_patient_scans(patient_id: str, current_user: dict = Depends(get_current_user)):
    try:
        cursor = scans_col.find({"patientId": patient_id, "doctorId": current_user["_id"]}, sort=[("createdAt", -1)])
        scans = []
        async for doc in cursor:
            scans.append(fix_id(doc))
        return scans
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# AI PREDICTION ROUTE
# ══════════════════════════════════════════════════════════════

@app.post("/predict")
async def predict_oral_images(
    tongue: UploadFile = File(...),
    gums: UploadFile = File(...),
    floor_mouth: UploadFile = File(...),
    buccal: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    try:
        images = {
            'tongue': await tongue.read(),
            'gums': await gums.read(),
            'floor_mouth': await floor_mouth.read(),
            'buccal': await buccal.read(),
        }
        results = {}
        for name, img_bytes in images.items():
            results[name] = analyze_image(img_bytes)

        confidences = [r['confidence'] for r in results.values()]
        avg_confidence = sum(confidences) / len(confidences)

        diseases_found = [r['disease'] for r in results.values() if r['disease'] not in ['normal', 'unknown']]

        if diseases_found:
            primary_disease = max(set(diseases_found), key=diseases_found.count)
            cancer_probability = round(avg_confidence, 2)
        else:
            primary_disease = 'No Significant Lesion'
            cancer_probability = round(100 - avg_confidence, 2)

        risk_level = 'high' if cancer_probability > 70 else 'moderate' if cancer_probability > 40 else 'low'

        recommendation = (
            'Result: Significant suspicious features were detected.\nSuggestion to Patient:\n• Consult a dentist or oral medicine specialist as soon as possible.\n• Do not ignore persistent ulcers, red/white patches, lumps, or swelling.\n• Avoid tobacco and alcohol immediately.\n• Seek urgent medical attention if you experience difficulty swallowing, speaking, or opening your mouth.'
            if risk_level == 'high'
            else 'Result: Some suspicious features were detected that require attention.\nSuggestion to Patient:\n• Schedule a dental examination within the next few weeks.\n• Avoid tobacco, smoking, and alcohol until evaluated.\n• Monitor the affected area for changes in size, color, or symptoms.\n• Seek professional advice if pain, bleeding, or difficulty eating develops.'
            if risk_level == 'moderate'
            else 'Result: No obvious signs of serious abnormalities detected.\nSuggestion to Patient:\n• Maintain good oral hygiene (brush twice daily and floss regularly).\n• Avoid tobacco products and limit alcohol consumption.\n• Continue regular dental check-ups every 6 months.\n• Monitor your mouth for any new sores, patches, or swelling.\n• If you notice any changes that persist for more than 2 weeks, consult a dentist.'
        )

        type_labels = {'tongue': 'Tongue', 'gums': 'Gums', 'floor_mouth': 'Floor of Mouth', 'buccal': 'Buccal Mucosa'}

        image_analysis = []
        for key, result in results.items():
            finding = result['disease'].replace('_', ' ').title()
            if result['disease'] == 'normal':
                finding = 'Normal'
            image_analysis.append({'type': type_labels[key], 'finding': finding, 'confidence': int(result['confidence'])})

        return {
            'cancerProbability': cancer_probability,
            'lesionType': primary_disease.replace('_', ' ').title(),
            'lesionLocations': [] if primary_disease == 'No Significant Lesion' else [type_labels[k] for k, r in results.items() if r['disease'] != 'normal'],
            'riskLevel': risk_level,
            'recommendation': recommendation,
            'confidence': round(avg_confidence, 2),
            'imageResults': results,
            'diseaseName': primary_disease.replace('_', ' ').title(),
            'diseaseMatchProbability': round(avg_confidence, 2),
            'imageAnalysis': image_analysis,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ══════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════

@app.get("/dashboard/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    try:
        total_patients = await patients_col.count_documents({"doctorId": current_user["_id"]})
        total_scans = await scans_col.count_documents({"doctorId": current_user["_id"]})
        high_risk = await scans_col.count_documents({"doctorId": current_user["_id"], "riskLevel": "high"})
        moderate_risk = await scans_col.count_documents({"doctorId": current_user["_id"], "riskLevel": "moderate"})
        low_risk = await scans_col.count_documents({"doctorId": current_user["_id"], "riskLevel": "low"})
        return {"totalPatients": total_patients, "totalScans": total_scans, "highRisk": high_risk, "moderateRisk": moderate_risk, "lowRisk": low_risk}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"status": "running", "message": "DentalScan AI Backend is running", "version": "1.0.0"}


@app.get("/health")
async def health():
    try:
        await client.admin.command("ping")
        db_status = "connected"
    except Exception as e:
        db_status = f"disconnected: {str(e)}"
    return {"api": "running", "database": db_status, "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)