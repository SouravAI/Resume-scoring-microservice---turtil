# Resume Scoring Service

A containerized microservice for scoring student resumes against specific career goals using machine learning and fuzzy matching. The service evaluates resumes for roles like Amazon Software Development Engineer (SDE), Machine Learning (ML) Internship, and GATE Electronics and Communication Engineering (ECE).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Docker Deployment](#docker-deployment)
- [API Endpoints](#api-endpoints)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview
This service scores resumes by analyzing their text against predefined goals using a trained Logistic Regression model and TF-IDF vectorization. It identifies matched and missing skills, provides grouped skill insights, and suggests learning paths for improvement. The service is built with FastAPI, uses NLTK for text processing, and is deployed via Docker for portability.

## Features
- **Resume Scoring**: Evaluates resumes for specific goals (Amazon SDE, ML Internship, GATE ECE) with a score (0.0 to 1.0).
- **Skill Matching**: Uses fuzzy matching and lemmatization to identify relevant skills, including alternate skill names.
- **Suggestions**: Provides learning path recommendations for missing skills.
- **Grouped Insights**: Organizes missing skills by category (e.g., Core CS, Backend & Infra).
- **Dockerized**: Fully containerized for easy deployment.
- **Offline Operation**: All models and resources are bundled for offline use.

## Project Structure
```
resume-scorer/
├── app/
│   ├── main.py              # FastAPI application
│   ├── scorer.py            # Resume scoring logic
│   └── model/               # Trained models and vectorizers
│       ├── amazon_sde_model.pkl
│       ├── amazon_sde_vectorizer.pkl
│       ├── ml_internship_model.pkl
│       ├── ml_internship_vectorizer.pkl
│       ├── gate_ece_model.pkl
│       ├── gate_ece_vectorizer.pkl
├── data/
│   ├── training_data/
│   │   ├── amazon_sde.json
│   │   ├── ml_internship.json
│   │   ├── gate_ece.json
│   ├── goals.json           # Goal-specific skills
│   ├── suggestions.json     # Learning path suggestions
│   ├── skill_groups.json    # Skill groupings
│   ├── alternate_skills.json # Alternate skill names
├── tests/
│   ├── test_score.py        # Unit tests for scoring logic
├── config.json              # Configuration settings
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
├── schema.json              # API request/response schema
├── README.md                # Project documentation
```

## Setup
### Prerequisites
- Python 3.10
- Docker Desktop (for containerized deployment)
- Git (optional, for cloning the repository)

### Local Setup
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd resume-scorer
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**:
   ```bash
   python -m nltk.downloader wordnet punkt averaged_perceptron_tagger
   ```

5. **Run the Application**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

6. **Access the API**:
   - Open `http://localhost:8000/` in a browser to verify the service.
   - Use the Swagger UI at `http://localhost:8000/docs` for interactive API testing.

## Docker Deployment
1. **Build the Docker Image**:
   ```bash
   docker build -t resume-scorer .
   ```

2. **Run the Container**:
   ```bash
   docker run -p 8000:8000 resume-scorer
   ```

3. **Test the API**:
   - Root endpoint: `http://localhost:8000/`
   - Health check: `http://localhost:8000/health`
   - Scoring endpoint: Send a POST request to `http://localhost:8000/score` (see below).

## API Endpoints
- **GET /**: Returns service version and status.
  - Response: `{"version":"1.0.0","status":"Resume Scoring Service Active"}`
- **GET /health**: Checks service health.
  - Response: `{"status":"ok"}`
- **POST /score**: Scores a resume against a specified goal.
  - Request Body:
    ```json
    {
      "student_id": "string",
      "goal": "Amazon SDE",
      "resume_text": "string"
    }
    ```
  - Response:
    ```json
    {
      "score": float,
      "is_pass": boolean,
      "matched_skills": [string],
      "missing_skills": [string],
      "missing_skills_grouped": {string: [string]},
      "suggested_learning_path": [string]
    }
    ```

## Usage
1. **Prepare a Resume**:
   - Create a text string containing the resume content, including skills and experiences.
   - Example: `"Proficient in Java, Python, Data Structures, and Algorithms. Built a REST API with Flask."`

2. **Send a Scoring Request**:
   - Use `curl`, Postman, or Swagger UI to send a POST request:
     ```bash
     curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{"student_id":"123","goal":"Amazon SDE","resume_text":"Proficient in Java, Python, Data Structures, and Algorithms"}'
     ```

3. **Interpret Results**:
   - `score`: Probability (0.0 to 1.0) of resume suitability.
   - `is_pass`: True if score ≥ 0.5 (configurable in `config.json`).
   - `matched_skills`: Skills found in the resume.
   - `missing_skills`: Skills not found, capped at 15.
   - `missing_skills_grouped`: Missing skills grouped by category (e.g., Core CS).
   - `suggested_learning_path`: Recommendations for missing skills.

## Testing
Run unit tests to validate the scoring logic:
```bash
python -m unittest tests/test_score.py
```

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.