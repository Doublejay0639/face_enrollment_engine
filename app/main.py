import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import database
from .routers import enrollmentsss


app = FastAPI()

database.client.admin.command("ping")


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(enrollmentsss.router)



@app.get("/")
def root():
    return {"message": "Welcome"}