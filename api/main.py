"""Minsy Backend API - Main Application."""

import platform
from contextlib import asynccontextmanager

import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.middleware import setup_middleware
from utils.logger import logger, banner, log_success, configure_uvicorn_logging

# Ensure uvicorn uses our logger
configure_uvicorn_logging()


def get_gpu_info() -> dict | None:
    """Get GPU usage info if available."""
    try:
        import subprocess

        # Try nvidia-smi for NVIDIA GPUs
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for i, line in enumerate(lines):
                parts = line.split(", ")
                if len(parts) == 3:
                    gpus.append({
                        "id": i,
                        "utilization_percent": float(parts[0]),
                        "memory_used_mb": float(parts[1]),
                        "memory_total_mb": float(parts[2]),
                    })
            return {"available": True, "gpus": gpus}
    except Exception:
        pass

    # Check for Apple Silicon GPU (Metal)
    if platform.system() == "Darwin" and platform.processor() == "arm":
        return {"available": True, "type": "Apple Silicon (Metal)", "note": "Detailed metrics not available"}

    return {"available": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    banner()
    log_success("Minsy Backend started successfully")
    logger.info(f"Running on Python {platform.python_version()}")
    yield
    # Shutdown
    logger.info("Minsy Backend shutting down...")


app = FastAPI(
    title="Minsy API",
    description="AI-powered quantitative trading strategy platform",
    version="0.1.0",
    lifespan=lifespan,
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup custom middleware
setup_middleware(app)

# Include routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    logger.debug("Root endpoint accessed")
    return {
        "name": "Minsy API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint with system resource usage."""
    logger.debug("Health check with system metrics requested")
    # CPU info
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()

    # Memory info
    memory = psutil.virtual_memory()

    # Disk info
    disk = psutil.disk_usage("/")

    return {
        "status": "healthy",
        "service": "minsy-backend",
        "system": {
            "platform": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "usage_percent": cpu_percent,
            "cores": cpu_count,
            "frequency_mhz": cpu_freq.current if cpu_freq else None,
        },
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "usage_percent": memory.percent,
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "usage_percent": round(disk.percent, 1),
        },
        "gpu": get_gpu_info(),
    }
