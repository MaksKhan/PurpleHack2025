from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from typing import Dict, Any
from ml import optimize_project

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProjectRequest(BaseModel):
    project_data: Dict[str, Any]

@app.post("/predict")
async def predict(request: ProjectRequest):
    try:
        # Конвертируем данные в JSON строку
        input_json = json.dumps(request.project_data)
        
        # Оптимизируем проект
        optimized_json = optimize_project(input_json)
        
        # Конвертируем результат обратно в словарь
        return json.loads(optimized_json)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Optimization error: {str(e)}"
        )
