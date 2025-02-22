from fastapi import FastAPI
from fastapi import File, UploadFile, FileResponse
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/send-and-rec")
async def send_and_rec(image: UploadFile = File(...)):
    
    return {FileResponse(ProcessImage(image))}