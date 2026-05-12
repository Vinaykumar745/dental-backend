from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import List
import motor.motor_asyncio
import hashlib
import os
from dotenv import load_dotenv
import uvicorn
import traceback

load_dotenv()

# ── App setup ─────────────────────────────────────────────────
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
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client.dental_db
users_col = db.get_collection("users")
patients_col = db.get_collection("patients")
scans_col = db.get_collection("scans")

# ── JWT setup ─────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "dental_scan_ai_secret_key_2025")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 10080  # 7 days

# ── Password hashing — using SHA256 to avoid bcrypt 72 byte limit ──
def hash_password(password: str) -> str:
    """Hash password using SHA256 — no byte limit issues"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    """Verify password by comparing SHA256 hashes"""
    return hashlib.sha256(plain.encode('utf-8')).hexdigest() == hashed

security = HTTPBearer()


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

class PatientRequest(BaseModel):
    id: str
    name: str
    age: int
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


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def create_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def fix_id(doc: dict) -> dict:
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

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
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please login again.",
        )
    user = await users_col.find_one({"email": email})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found.")
    return fix_id(user)


# ══════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/auth/signup")
async def signup(req: SignUpRequest):
    try:
        # Validate
        if not req.name or len(req.name.strip()) < 2:
            raise HTTPException(status_code=400, detail="Name must be at least 2 characters.")
        if not req.email or "@" not in req.email:
            raise HTTPException(status_code=400, detail="Please enter a valid email address.")
        if not req.password or len(req.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

        # Check duplicate email
        existing = await users_col.find_one({"email": req.email.lower().strip()})
        if existing:
            raise HTTPException(
                status_code=400,
                detail="An account with this email already exists. Please login instead.",
            )

        # Create user with SHA256 hashed password
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

        print(f"✅ New user signed up: {user_doc['email']}")

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": user_doc["_id"],
                "name": user_doc["name"],
                "email": user_doc["email"],
                "role": user_doc["role"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")


@app.post("/auth/login")
async def login(req: LoginRequest):
    try:
        if not req.email or "@" not in req.email:
            raise HTTPException(status_code=400, detail="Please enter a valid email address.")
        if not req.password:
            raise HTTPException(status_code=400, detail="Please enter your password.")

        # Find user
        user = await users_col.find_one({"email": req.email.lower().strip()})
        if not user:
            raise HTTPException(
                status_code=401,
                detail="No account found with this email. Please sign up first.",
            )

        # Verify password
        if not verify_password(req.password, user["password"]):
            raise HTTPException(
                status_code=401,
                detail="Invalid Password. Please try again.",
            )

        user = fix_id(user)
        token = create_token({"sub": user["email"]})

        print(f"✅ User logged in: {user['email']}")

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": user["_id"],
                "name": user["name"],
                "email": user["email"],
                "role": user.get("role", "doctor"),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["_id"],
        "name": current_user["name"],
        "email": current_user["email"],
        "role": current_user.get("role", "doctor"),
    }


# ══════════════════════════════════════════════════════════════
# PATIENT ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/patients")
async def create_patient(
    req: PatientRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        patient_doc = {
            "id": req.id,
            "doctorId": current_user["_id"],
            "name": req.name.strip(),
            "age": req.age,
            "date": req.date,
            "mobile": req.mobile,
            "createdAt": req.createdAt,
            "updatedAt": datetime.utcnow().isoformat(),
        }
        existing = await patients_col.find_one(
            {"id": req.id, "doctorId": current_user["_id"]}
        )
        if existing:
            await patients_col.update_one(
                {"id": req.id, "doctorId": current_user["_id"]},
                {"$set": patient_doc},
            )
        else:
            await patients_col.insert_one(patient_doc)
        return {"message": "Patient saved successfully", "patientId": req.id}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save patient: {str(e)}")


@app.get("/patients")
async def get_patients(current_user: dict = Depends(get_current_user)):
    try:
        cursor = patients_col.find(
            {"doctorId": current_user["_id"]}, sort=[("createdAt", -1)]
        )
        patients = []
        async for doc in cursor:
            patients.append(fix_id(doc))
        return patients
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients/{patient_id}")
async def get_patient(
    patient_id: str,
    current_user: dict = Depends(get_current_user),
):
    patient = await patients_col.find_one(
        {"id": patient_id, "doctorId": current_user["_id"]}
    )
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return fix_id(patient)


# ══════════════════════════════════════════════════════════════
# SCAN ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/scans")
async def save_scan(
    req: ScanResultRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        scan_doc = {
            "patientId": req.patientId,
            "doctorId": current_user["_id"],
            "cancerProbability": req.cancerProbability,
            "lesionType": req.lesionType,
            "lesionLocations": req.lesionLocations,
            "riskLevel": req.riskLevel,
            "recommendation": req.recommendation,
            "imageAnalysis": [a.dict() for a in req.imageAnalysis],
            "scanDate": req.scanDate,
            "createdAt": datetime.utcnow().isoformat(),
        }
        result = await scans_col.insert_one(scan_doc)
        await patients_col.update_one(
            {"id": req.patientId, "doctorId": current_user["_id"]},
            {
                "$set": {
                    "latestRisk": req.riskLevel,
                    "latestProbability": req.cancerProbability,
                    "latestLesion": req.lesionType,
                    "lastScanDate": req.scanDate,
                }
            },
        )
        return {"message": "Scan saved successfully", "scanId": str(result.inserted_id)}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save scan: {str(e)}")


@app.get("/scans")
async def get_all_scans(current_user: dict = Depends(get_current_user)):
    try:
        cursor = scans_col.find(
            {"doctorId": current_user["_id"]}, sort=[("createdAt", -1)]
        )
        scans = []
        async for doc in cursor:
            scans.append(fix_id(doc))
        return scans
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scans/{patient_id}")
async def get_patient_scans(
    patient_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        cursor = scans_col.find(
            {"patientId": patient_id, "doctorId": current_user["_id"]},
            sort=[("createdAt", -1)],
        )
        scans = []
        async for doc in cursor:
            scans.append(fix_id(doc))
        return scans
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        return {
            "totalPatients": total_patients,
            "totalScans": total_scans,
            "highRisk": high_risk,
            "moderateRisk": moderate_risk,
            "lowRisk": low_risk,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "DentalScan AI Backend is running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    try:
        await client.admin.command("ping")
        db_status = "connected"
    except Exception as e:
        db_status = f"disconnected: {str(e)}"
    return {
        "api": "running",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
