import json
import os
import joblib
from typing import Dict, Any, List, Tuple
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from fuzzywuzzy import fuzz
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def get_project_path(filename: str) -> str:
    """Compute file path relative to project root."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(project_root, "data", filename)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text_for_matching(text):
    """Tokenize and lemmatize text, return as a string."""
    tokens = nltk.word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    return ' '.join(lemmatized)

def phrase_in_lemmatized_text(phrase, lemmatized_text, threshold=70, config=None):
    """Check if phrase exists in lemmatized text using fuzzy matching."""
    phrase_lemmatized = lemmatize_text_for_matching(phrase)
    phrase_words = phrase_lemmatized.split()
    n = len(phrase_words)
    text_words = lemmatized_text.split()
    max_score = 0
    for i in range(len(text_words) - n + 1):
        candidate = ' '.join(text_words[i:i+n])
        score = fuzz.ratio(candidate, phrase_lemmatized)
        max_score = max(max_score, score)
        if score >= threshold:
            return True
    # Partial n-gram matching for multi-word skills
    if n > 1:
        for word in phrase_words:
            for text_word in text_words:
                if fuzz.ratio(word, text_word) >= 80:  # Lowered from 85
                    return True
    if config and config.get("log_score_details", False):
        logger.info(f"Skill '{phrase}' max fuzzy ratio: {max_score} (threshold={threshold})")
    return False

def load_skill_groups() -> Dict[str, list]:
    """Load skill groups from JSON file."""
    path = get_project_path("skill_groups.json")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading skill_groups.json: {str(e)}")
        return {}

def load_alternate_skills() -> Dict[str, list]:
    """Load alternate skills from JSON file."""
    path = get_project_path("alternate_skills.json")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading alternate_skills.json: {str(e)}")
        return {}

def load_model(goal: str):
    """Load trained model for the specified goal."""
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    filename = f"{goal.replace(' ', '_')}_model.pkl"
    model_path = os.path.join(model_dir, filename)
    try:
        return joblib.load(model_path)
    except FileNotFoundError as e:
        logger.error(f"Error loading model {model_path}: {str(e)}")
        raise

def load_vectorizer(goal: str):
    """Load vectorizer for the specified goal."""
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    filename = f"{goal.replace(' ', '_')}_vectorizer.pkl"
    model_path = os.path.join(model_dir, filename)
    try:
        return joblib.load(model_path)
    except FileNotFoundError as e:
        logger.error(f"Error loading vectorizer {model_path}: {str(e)}")
        raise

def score_resume(student_id: str, goal: str, resume_text: str, config: Dict[str, Any], goals_map: Dict, suggestion_map: Dict, skill_groups: Dict):
    """Score a resume against a goal, returning matched/missing skills and suggestions."""
    # Validate input
    if not resume_text.strip():
        logger.error(f"Empty resume text for student_id: {student_id}")
        raise ValueError("Resume text cannot be empty")

    # Load model and compute model score
    try:
        model = load_model(goal)
        vectorizer = load_vectorizer(goal)
        X_transformed = vectorizer.transform([resume_text])
        prob = model.predict_proba(X_transformed)[0][1]
    except Exception as e:
        logger.error(f"Error scoring resume for student_id {student_id}: {str(e)}")
        raise

    # Extract goal-specific skills
    goal_skills_with_importance = {item["name"]: item["importance"] for item in goals_map.get(goal, [])}
    goal_skill_names = set(goal_skills_with_importance.keys())
    importance_order = {"core": 1, "important": 2, "nice_to_have": 3}

    # Load alternate skills
    alternate_skills_map = load_alternate_skills()

    # Preprocess resume text
    lemmatized_resume_text = lemmatize_text_for_matching(resume_text)
    if config.get("log_score_details", False):
        logger.info(f"lemmatized_resume_text='{lemmatized_resume_text}'")

    # Match skills using fuzzy phrase matching
    matched_skills = []
    all_missing_skills = []
    for skill_name in goal_skill_names:
        confidence_threshold = 70  # Lowered from 75
        partial_threshold = 55   # Lowered from 60
        if phrase_in_lemmatized_text(skill_name, lemmatized_resume_text, threshold=confidence_threshold, config=config):
            matched_skills.append(skill_name)
            if config.get("log_score_details", False):
                logger.info(f"Matched skill '{skill_name}' directly")
        else:
            found_alternate = False
            if skill_name in alternate_skills_map:
                for alt in alternate_skills_map[skill_name]:
                    if phrase_in_lemmatized_text(alt, lemmatized_resume_text, threshold=confidence_threshold, config=config):
                        matched_skills.append(skill_name)
                        found_alternate = True
                        if config.get("log_score_details", False):
                            logger.info(f"Matched skill '{skill_name}' via alternate '{alt}'")
                        break
            if not found_alternate:
                if phrase_in_lemmatized_text(skill_name, lemmatized_resume_text, threshold=partial_threshold, config=config):
                    matched_skills.append(skill_name)
                    if config.get("log_score_details", False):
                        logger.info(f"Partially matched skill '{skill_name}' (fuzzy)")
                else:
                    importance = goal_skills_with_importance.get(skill_name, "unknown")
                    all_missing_skills.append((skill_name, importance))
                    if config.get("log_score_details", False):
                        logger.info(f"Missing skill '{skill_name}' (Importance: {importance})")

    # Calculate skill score
    importance_weights = {"core": 3, "important": 2, "nice_to_have": 1}
    total_skill_points = sum(
        importance_weights.get(goal_skills_with_importance.get(skill, "nice_to_have"), 0)
        for skill in goal_skill_names
    )
    matched_skill_points = sum(
        importance_weights.get(goal_skills_with_importance.get(skill, "nice_to_have"), 0)
        for skill in matched_skills
    )
    skill_score = matched_skill_points / total_skill_points if total_skill_points > 0 else 0

    # Combine with model score
    model_weight = 0.6  # Reduced from 0.7
    skill_weight = 0.4  # Increased from 0.3
    final_score = model_weight * prob + skill_weight * skill_score

    # Determine pass/fail based on final score
    is_pass = bool(final_score >= config["minimum_score_to_pass"])

    # Cap missing skills based on config
    max_missing_skills = config.get("max_missing_skills", 15)
    sorted_missing_skills = sorted(all_missing_skills, key=lambda x: (importance_order.get(x[1], 99), x[0]))
    capped_missing_skills_names = [skill_name for skill_name, _ in sorted_missing_skills][:max_missing_skills]

    # Generate suggestions
    suggestions = [suggestion_map.get(skill_name, f"Learn basics of {skill_name}") for skill_name in capped_missing_skills_names]

    # Group-level insights
    missing_grouped = {}
    for group, skills_in_group in skill_groups.get(goal, {}).items():
        group_skill_names = {item["name"] for item in goals_map.get(goal, []) if item["name"] in skills_in_group}
        group_matched = [s for s in matched_skills if s in group_skill_names]
        if group_matched:
            to_recommend = sorted([s for s in group_skill_names if s not in matched_skills])
            if to_recommend:
                missing_grouped[group] = to_recommend
        else:
            group_missing = [skill_name for skill_name, _ in all_missing_skills if skill_name in group_skill_names]
            if group_missing:
                missing_grouped[group] = sorted(group_missing)

    # Log summary
    if config.get("log_score_details", False):
        logger.info(f"matched_skills={matched_skills}")
        logger.info(f"missing_skills={capped_missing_skills_names}")
        logger.info(f"suggested_learning_path={suggestions}")
        logger.info(
            f"Model score: {prob:.4f}, Skill score: {skill_score:.4f}, "
            f"Final score: {final_score:.4f} — {'Pass' if is_pass else 'Fail'}"
        )

    return {
        "score": final_score,
        "is_pass": is_pass,
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(capped_missing_skills_names),
        "missing_skills_grouped": missing_grouped,
        "suggested_learning_path": suggestions
    }