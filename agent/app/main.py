from fastapi import FastAPI
from app.routers.app_routers import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Backend API of the Agent",
        "docs": "/docs",
        "api_base": "/api"
    }

app.include_router(router, prefix="/api", tags=["api"])