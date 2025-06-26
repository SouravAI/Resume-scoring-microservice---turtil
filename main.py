import json
import os
from typing import Dict, List
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from app.scorer import score_resume

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI instance with metadata
app = FastAPI(
    title="Resume Scoring Microservice",
    description="Containerized microservice for scoring student resumes offline",
    version="1.0.0"
)

# Utility function for file path computation
def get_project_path(filename: str) -> str:
    """Compute file path relative to project root."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(project_root, "data", filename)

# Load and validate config and JSON files at startup
try:
    config = json.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"), "r"))
    if "model_goals_supported" not in config:
        raise ValueError("Missing 'model_goals_supported' in config.json")
    goals_map = json.load(open(get_project_path("goals.json"), "r"))
    suggestion_map = json.load(open(get_project_path("suggestions.json"), "r"))
    skill_groups = json.load(open(get_project_path("skill_groups.json"), "r"))
except FileNotFoundError as e:
    logger.error(f"Error loading configuration file: {str(e)}")
    raise
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in configuration file: {str(e)}")
    raise

# Pydantic models for request/response validation
class ScoreRequest(BaseModel):
    student_id: str
    goal: str
    resume_text: str

class ScoreResponse(BaseModel):
    score: float
    is_pass: bool
    matched_skills: list[str]
    missing_skills: list[str]
    missing_skills_grouped: Dict[str, List[str]]
    suggested_learning_path: list[str]

# Version & health endpoints
@app.get("/", tags=["Metadata"])
def root():
    logger.info("Root endpoint accessed")
    return {"version": app.version, "status": "Resume Scoring Service Active"}

@app.get("/health", tags=["Metadata"])
def health():
    logger.info("Health check endpoint accessed")
    return {"status": "ok"}

# Main scoring endpoint
@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def post_score(request: ScoreRequest):
    logger.info(f"Scoring request received for student_id: {request.student_id}, goal: {request.goal}")
    # Input validation
    if not request.resume_text.strip():
        logger.warning(f"Empty resume text for student_id: {request.student_id}")
        raise HTTPException(status_code=400, detail="Resume text cannot be empty")
    # Unsupported goal check
    if request.goal not in config["model_goals_supported"]:
        logger.warning(f"Unsupported goal requested: {request.goal}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported goal '{request.goal}'. Supported: {config['model_goals_supported']}"
        )
    # Call scorer logic with error handling
    try:
        result = score_resume(
            student_id=request.student_id,
            goal=request.goal,
            resume_text=request.resume_text,
            config=config,
            goals_map=goals_map,
            suggestion_map=suggestion_map,
            skill_groups=skill_groups
        )
        logger.info(f"Scoring completed for student_id: {request.student_id}")
        return result
    except ValueError as e:
        logger.error(f"Validation error for student_id {request.student_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing resume for student_id {request.student_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")