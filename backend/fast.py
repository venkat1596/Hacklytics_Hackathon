from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-production-url.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/send-and-rec")
async def send_and_rec(image: UploadFile = File(...)):
    try:
        file_path = f"{UPLOAD_FOLDER}/{image.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        return {"filename": image.filename, "message": "Upload successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
