import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean resume text by converting to lowercase and preserving technical terms like C++, RESTful."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9+#./ ]', ' ', text)  # Keep +, #, dots, slashes for skills
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data(file_path):
    """Load data from a JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        return None

# Load config
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    supported_goals = config['model_goals_supported']
except FileNotFoundError:
    logger.error("config.json not found")
    exit(1)
except json.JSONDecodeError:
    logger.error("Invalid config.json")
    exit(1)
except KeyError:
    logger.error("Missing 'model_goals_supported' in config.json")
    exit(1)

# Mapping of goals to their respective data files
goal_to_file = {
    "Amazon SDE": "amazon_sde.json",
    "GATE ECE": "gate_ece.json",
    "ML Internship": "ml_internship.json"
}

for goal in supported_goals:
    logger.info(f"Training model for goal: {goal}")
    file_name = goal_to_file.get(goal)
    if not file_name:
        logger.error(f"No data file mapped for goal: {goal}")
        continue
    
    file_path = f'data/training_data/{file_name}'
    data = load_data(file_path)
    if data is None:
        continue
    
    # Extract and clean texts and labels
    texts = [clean_text(item['resume_text']) for item in data]
    labels = [int(item['label']) for item in data]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)
    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))  # Increased from 1000
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Logistic Regression model
    model = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=200)  # Lower C, increase max_iter
    model.fit(X_train_vec, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy for {goal}: {accuracy:.4f}")
    logger.info(f"Classification Report for {goal}:\n{classification_report(y_test, y_pred)}")
    
    # Save the vectorizer and model
    model_dir = 'app/model/'
    goal_key = goal.replace(" ", "_").lower()
    joblib.dump(vectorizer, f'{model_dir}{goal_key}_vectorizer.pkl')
    joblib.dump(model, f'{model_dir}{goal_key}_model.pkl')
    logger.info(f"Model and vectorizer saved for {goal}")