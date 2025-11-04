from fastapi import FastAPI, HTTPException, Depends, UploadFile
from fastapi.security import APIKeyHeader
from src.utils.config import APP_NAME, VERSION, API_SECRET_KEY
from src.utils.schemas import PredictionResponse
from src.inference import classify_image

# Initialize an app
app = FastAPI(title=APP_NAME, version=VERSION)


api_key_header = APIKeyHeader(name='X-API-Key')
async def verify_api_key(api_key: str=Depends(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="You are not authorized to use this API")
    return api_key


@app.get('/', tags=['check'])
async def home(api_key: str=Depends(verify_api_key)):
    return {
        "app_name": APP_NAME,
        "version": VERSION,
        "status": "up & running"
    }



@app.post("/classify", tags=['CNN'], response_model=PredictionResponse)
async def classify(file: UploadFile, api_key: str=Depends(verify_api_key)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
            
        contents = await file.read()
        response = classify_image(contents)
        return PredictionResponse(**response)
    
    except Exception as e:
        raise HTTPException(500, f"Error making predictions: {str(e)}")
