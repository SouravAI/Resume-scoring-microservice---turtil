import unittest
import os
import json
from unittest.mock import patch
from app.scorer import score_resume, load_model, load_vectorizer
from fastapi.testclient import TestClient
from app.main import app

class TestResumeScorer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures, including config and JSON data."""
        # Load configuration and data files
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(self.project_root, "config.json")) as f:
            self.config = json.load(f)
        with open(os.path.join(self.project_root, "data", "goals.json")) as f:
            self.goals_map = json.load(f)
        with open(os.path.join(self.project_root, "data", "suggestions.json")) as f:
            self.suggestion_map = json.load(f)
        with open(os.path.join(self.project_root, "data", "skill_groups.json")) as f:
            self.skill_groups = json.load(f)
        
        # Sample resume text for testing
        self.sample_resume = "Proficient in Java, Python, Data Structures, Algorithms, SQL"
        self.student_id = "123"
        self.goal = "Amazon SDE"
        
        # Initialize FastAPI test client
        self.client = TestClient(app)

    def test_score_resume_valid_input(self):
        """Test score_resume with valid input."""
        result = score_resume(
            student_id=self.student_id,
            goal=self.goal,
            resume_text=self.sample_resume,
            config=self.config,
            goals_map=self.goals_map,
            suggestion_map=self.suggestion_map,
            skill_groups=self.skill_groups
        )
        
        # Validate response structure
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
        self.assertIn("is_pass", result)
        self.assertIn("matched_skills", result)
        self.assertIn("missing_skills", result)
        self.assertIn("missing_skills_grouped", result)
        self.assertIn("suggested_learning_path", result)
        
        # Validate types and constraints
        self.assertIsInstance(result["score"], float)
        self.assertTrue(0.0 <= result["score"] <= 1.0)
        self.assertIsInstance(result["is_pass"], bool)
        self.assertIsInstance(result["matched_skills"], list)
        self.assertIsInstance(result["missing_skills"], list)
        self.assertIsInstance(result["missing_skills_grouped"], dict)
        self.assertIsInstance(result["suggested_learning_path"], list)
        
        # Validate specific skills
        expected_matched = ["Java", "Python", "Data Structures", "Algorithms", "SQL"]
        for skill in expected_matched:
            self.assertIn(skill, result["matched_skills"])
        
        # Validate score threshold
        self.assertEqual(result["is_pass"], result["score"] >= self.config["minimum_score_to_pass"])

    def test_score_resume_empty_resume_text(self):
        """Test score_resume with empty resume text."""
        with self.assertRaises(ValueError) as context:
            score_resume(
                student_id=self.student_id,
                goal=self.goal,
                resume_text="",
                config=self.config,
                goals_map=self.goals_map,
                suggestion_map=self.suggestion_map,
                skill_groups=self.skill_groups
            )
        self.assertEqual(str(context.exception), "Resume text cannot be empty")

    def test_score_resume_invalid_goal(self):
        """Test score_resume with unsupported goal."""
        with self.assertRaises(FileNotFoundError):
            score_resume(
                student_id=self.student_id,
                goal="Invalid Goal",
                resume_text=self.sample_resume,
                config=self.config,
                goals_map=self.goals_map,
                suggestion_map=self.suggestion_map,
                skill_groups=self.skill_groups
            )

    @patch('app.scorer.load_model')
    @patch('app.scorer.load_vectorizer')
    def test_score_resume_model_prediction(self, mock_vectorizer, mock_model):
        """Test score_resume with mocked model and vectorizer."""
        # Mock model and vectorizer
        mock_model.return_value.predict_proba.return_value = [[0.2, 0.8]]  # 80% probability for positive class
        mock_vectorizer.return_value.transform.return_value = [[]]  # Dummy vectorized input
        
        result = score_resume(
            student_id=self.student_id,
            goal=self.goal,
            resume_text=self.sample_resume,
            config=self.config,
            goals_map=self.goals_map,
            suggestion_map=self.suggestion_map,
            skill_groups=self.skill_groups
        )
        
        self.assertAlmostEqual(result["score"], 0.8)
        self.assertTrue(result["is_pass"])  # 0.8 >= 0.5 (minimum_score_to_pass)

    def test_api_score_endpoint(self):
        """Test the /score endpoint via FastAPI TestClient."""
        response = self.client.post(
            "/score",
            json={
                "student_id": self.student_id,
                "goal": self.goal,
                "resume_text": self.sample_resume
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Validate response structure
        self.assertIn("score", data)
        self.assertIn("is_pass", data)
        self.assertIn("matched_skills", data)
        self.assertIn("missing_skills", data)
        self.assertIn("missing_skills_grouped", data)
        self.assertIn("suggested_learning_path", data)
        
        # Validate specific skills
        expected_matched = ["Java", "Python", "Data Structures", "Algorithms", "SQL"]
        for skill in expected_matched:
            self.assertIn(skill, data["matched_skills"])

    def test_api_score_empty_resume_text(self):
        """Test /score endpoint with empty resume text."""
        response = self.client.post(
            "/score",
            json={
                "student_id": self.student_id,
                "goal": self.goal,
                "resume_text": ""
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Resume text cannot be empty")

    def test_api_score_invalid_goal(self):
        """Test /score endpoint with unsupported goal."""
        response = self.client.post(
            "/score",
            json={
                "student_id": self.student_id,
                "goal": "Invalid Goal",
                "resume_text": self.sample_resume
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported goal 'Invalid Goal'", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()