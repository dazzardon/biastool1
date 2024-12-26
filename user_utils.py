import os
import json
import hashlib
import logging

# --- Define Base Directory ---
BASE_DIR = r"C:\Users\Darren\Desktop\MediaPropaganda"

# Paths to user data and history
user_data_path = os.path.join(BASE_DIR, "users.json")
history_path = os.path.join(BASE_DIR, "history.json")

# Configure Logging
logger = logging.getLogger(__name__)

def hash_password(password):
    """
    Hash the password using SHA-256.
    """
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, name, email, password):
    """
    Create a new user and save to the users.json file.
    """
    try:
        if os.path.isfile(user_data_path):
            with open(user_data_path, 'r') as f:
                users = json.load(f)
        else:
            users = {}

        if username in users or any(u['email'] == email for u in users.values()):
            return False  # Username or email already exists

        users[username] = {
            'name': name,
            'email': email,
            'password': hash_password(password)
        }

        with open(user_data_path, 'w') as f:
            json.dump(users, f, indent=4)

        logger.info(f"User '{username}' created successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating user '{username}': {e}")
        return False

def get_user(username):
    """
    Retrieve user information by username.
    """
    try:
        if not os.path.isfile(user_data_path):
            return None

        with open(user_data_path, 'r') as f:
            users = json.load(f)

        return users.get(username, None)
    except Exception as e:
        logger.error(f"Error retrieving user '{username}': {e}")
        return None

def verify_password(username, password):
    """
    Verify the user's password.
    """
    try:
        user = get_user(username)
        if not user:
            return False
        return user['password'] == hash_password(password)
    except Exception as e:
        logger.error(f"Error verifying password for user '{username}': {e}")
        return False

def reset_password(username, new_password):
    """
    Reset the user's password.
    """
    try:
        if not os.path.isfile(user_data_path):
            return False

        with open(user_data_path, 'r') as f:
            users = json.load(f)

        if username not in users:
            return False

        users[username]['password'] = hash_password(new_password)

        with open(user_data_path, 'w') as f:
            json.dump(users, f, indent=4)

        logger.info(f"Password for user '{username}' has been reset.")
        return True
    except Exception as e:
        logger.error(f"Error resetting password for user '{username}': {e}")
        return False

def load_default_bias_terms():
    """
    Load default bias terms.
    """
    return ["biased", "manipulative", "deceptive"]  # Add more default terms as needed

def load_custom_bias_terms(user_terms):
    """
    Load custom bias terms provided by the user.
    """
    try:
        # Combine default and custom terms, ensuring no duplicates
        default_terms = load_default_bias_terms()
        combined_terms = list(set(default_terms + user_terms))
        return combined_terms
    except Exception as e:
        logger.error(f"Error loading custom bias terms: {e}")
        return load_default_bias_terms()

def save_analysis_to_history(analysis):
    """
    Save the analysis result to the user's history.
    """
    try:
        if os.path.isfile(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = {}

        user_email = analysis.get('email', 'guest')
        if user_email not in history:
            history[user_email] = []

        history[user_email].append(analysis)

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

        logger.info(f"Analysis saved to history for user '{user_email}'.")
    except Exception as e:
        logger.error(f"Error saving analysis to history for user '{analysis.get('username', 'guest')}': {e}")

def load_user_history(email):
    """
    Load the user's analysis history.
    """
    try:
        if not os.path.isfile(history_path):
            return []

        with open(history_path, 'r') as f:
            history = json.load(f)

        return history.get(email, [])
    except Exception as e:
        logger.error(f"Error loading history for email '{email}': {e}")
        return []
