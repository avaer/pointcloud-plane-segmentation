import os
import json
import subprocess
from typing import List

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Define data models
class Plane(BaseModel):
    normal: List[float]
    center: List[float]
    basisU: List[float]
    basisV: List[float]
    distanceFromOrigin: float
    inlierCount: int

# Create FastAPI app
app = FastAPI(title="Plane Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Path to the C++ executable
EXECUTABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "stdin_planes")

@app.get("/")
async def read_root():
    return {"message": "Plane Detection API is running"}

@app.post("/planes", response_model=List[Plane])
async def detect_planes(
    request: Request,
    width: int = Query(..., description="Width of the point cloud grid"),
    height: int = Query(..., description="Height of the point cloud grid"),
    min_normal_diff: float = Query(None, description="Minimum normal difference in degrees (default: 60)"),
    max_dist: float = Query(None, description="Maximum distance in degrees (default: 75)"),
    outlier_ratio: float = Query(None, description="Maximum outlier ratio (default: 0.75)"),
    min_num_points: int = Query(None, description="Minimum number of points (default: 30)"),
    nr_neighbors: int = Query(None, description="Number of neighbors for KNN (default: 75)")
):
    # Get raw query parameters to check which ones were actually provided
    query_params = dict(request.query_params)
    param_provided = {
        "min_normal_diff": "min_normal_diff" in query_params,
        "max_dist": "max_dist" in query_params,
        "outlier_ratio": "outlier_ratio" in query_params,
        "min_num_points": "min_num_points" in query_params,
        "nr_neighbors": "nr_neighbors" in query_params
    }
    # Read raw binary data from request
    binary_data = await request.body()
    
    # Expected size: 4 bytes (f32) * 3 coordinates * width * height
    expected_bytes = 4 * 3 * width * height
    if len(binary_data) != expected_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Binary data size mismatch: got {len(binary_data)} bytes, expected {expected_bytes} bytes"
        )
    
    # Check if executable exists
    if not os.path.exists(EXECUTABLE_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"C++ executable not found at {EXECUTABLE_PATH}. Make sure to build the project first."
        )
    
    try:
        # Prepare command with optional parameters
        command = [EXECUTABLE_PATH, str(width), str(height)]
        
        # Add optional parameters only if they were explicitly provided in the query string
        if param_provided["min_normal_diff"] and min_normal_diff is not None:
            command.extend(["--min-normal-diff", str(min_normal_diff)])
            
        if param_provided["max_dist"] and max_dist is not None:
            command.extend(["--max-dist", str(max_dist)])
            
        if param_provided["outlier_ratio"] and outlier_ratio is not None:
            command.extend(["--outlier-ratio", str(outlier_ratio)])
            
        if param_provided["min_num_points"] and min_num_points is not None:
            command.extend(["--min-num-points", str(min_num_points)])
            
        if param_provided["nr_neighbors"] and nr_neighbors is not None:
            command.extend(["--nr-neighbors", str(nr_neighbors)])
        
        # Run the C++ program and pass the binary data directly
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Send binary input and get output
        stdout, stderr = process.communicate(input=binary_data)
        
        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"C++ process failed with code {process.returncode}: {stderr.decode()}"
            )
        
        # Parse JSON output
        try:
            planes = json.loads(stdout.decode())
            return planes
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse JSON output: {e}. Output was: {stdout.decode()[:200]}..."
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process request: {str(e)}"
        )

# Run with: uvicorn server:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8555, reload=True)