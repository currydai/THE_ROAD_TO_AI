from fastapi import FastAPI

from agent_from_zero.api.routers import router

app = FastAPI(title="Agent From Zero API", version="0.1.0")
app.include_router(router)
