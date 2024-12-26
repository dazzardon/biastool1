# media_bias_detection.py

import streamlit as st
import logging
import datetime
import os
import json
import pandas as pd
import requests
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import unicodedata
import ssl

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- FastAPI Backend Configuration ---
API_BASE_URL = "http://localhost:8000/api"  # Update this if your FastAPI is hosted elsewhere

# --- Helper Functions ---
def is_strong_password(password):
    """Check if the password is strong."""
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    return True

def is_valid_email(email):
    """Validate email format."""
    return re.match(r'^[^@]+@[^@]+\.[^@]+$', email) is not None

def is_valid_url(url):
    """Validate URL format."""
    parsed = urlparse(url)
    return all([parsed.scheme, parsed.netloc])

def sanitize_key(s):
    """Sanitize string to be used as a key by replacing non-alphanumeric characters with underscores."""
    return re.sub(r'\W+', '_', s)

async def fetch_article_text_async(url):
    """Asynchronously fetch article text from a URL."""
    if not is_valid_url(url):
        st.error("Invalid URL format.")
        return None
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, ssl=ssl_context, timeout=10) as response:
                if response.status != 200:
                    st.error(f"HTTP Error: {response.status}")
                    return None
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                main_content = soup.find('main')
                if main_content:
                    article_text = main_content.get_text(separator=' ', strip=True)
                else:
                    paragraphs = soup.find_all('p')
                    article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
                if not article_text.strip():
                    st.error("No content found at the provided URL.")
                    return None
                return article_text
    except Exception as e:
        st.error(f"Error fetching the article: {e}")
        logger.error(f"Error fetching the article: {e}")
        return None

def fetch_article_text(url):
    """Fetch article text by running the asynchronous fetch function."""
    try:
        article_text = asyncio.run(fetch_article_text_async(url))
        return article_text
    except Exception as e:
        st.error(f"Error in fetching article text: {e}")
        logger.error(f"Error in fetching article text: {e}")
        return None

def preprocess_text(text):
    """Preprocess text by normalizing and removing non-ASCII characters."""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

def load_default_bias_terms():
    """Load default bias terms."""
    bias_terms = [
        'alarming', 'allegations', 'unfit', 'aggressive', 'alleged', 'apparently', 'arguably',
        'claims', 'controversial', 'disputed', 'insists', 'questionable', 'reportedly', 'rumored',
        'suggests', 'supposedly', 'unconfirmed', 'suspected', 'reckless', 'radical', 'extremist',
        'biased', 'manipulative', 'deceptive', 'unbelievable', 'incredible', 'shocking', 'outrageous',
        'bizarre', 'absurd', 'ridiculous', 'disgraceful', 'disgusting', 'horrible', 'terrible',
        'unacceptable', 'unfair', 'scandalous', 'suspicious', 'illegal', 'illegitimate', 'immoral',
        'corrupt', 'criminal', 'dangerous', 'threatening', 'harmful', 'menacing', 'disturbing',
        'distressing', 'troubling', 'fearful', 'afraid', 'panic', 'terror', 'catastrophe', 'disaster',
        'chaos', 'crisis', 'collapse', 'failure', 'ruin', 'devastation', 'suffering', 'misery', 'pain',
        'dreadful', 'awful', 'nasty', 'vile', 'vicious', 'brutal', 'violent', 'greedy', 'selfish',
        'arrogant', 'ignorant', 'stupid', 'unwise', 'illogical', 'unreasonable', 'delusional',
        'paranoid', 'obsessed', 'fanatical', 'zealous', 'militant', 'dictator', 'regime'
    ]
    bias_terms = list(set([term.lower() for term in bias_terms]))
    return bias_terms

def save_analysis_to_history(data):
    """Save analysis data to history for logged-in users."""
    email = st.session_state.get('email', 'guest')
    if email == 'guest':
        return  # Do not save for guests

    history_file = f"{email}_history.json"
    try:
        # Load existing history or initialize a new list
        history = []
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

        # Append the new analysis
        history.append(data)

        # Save back to the file
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
        logger.info(f"Analysis saved to {history_file}.")
    except Exception as e:
        logger.error(f"Error saving analysis to history: {e}")
        st.error("Failed to save analysis history. Check logs for details.")

def load_user_history(email):
    """Load analysis history for a user."""
    history_file = f"{email}_history.json"
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return history
        else:
            return []
    except Exception as e:
        logger.error(f"Error loading user history: {e}")
        st.error("Failed to load analysis history. Check logs for details.")
        return []

def interpret_confidence(score):
    """Interpret confidence score into qualitative labels."""
    if score > 0.75:
        return "High Confidence"
    elif score > 0.5:
        return "Moderate Confidence"
    else:
        return "Low Confidence"

def get_bias_terms_via_api():
    """Fetch bias terms from the FastAPI backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/bias-terms")
        if response.status_code == 200:
            return response.json().get("bias_terms", [])
        else:
            st.error(f"Failed to fetch bias terms: {response.status_code}")
            logger.error(f"Bias terms API error: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching bias terms: {e}")
        logger.error(f"Error fetching bias terms: {e}")
        return []

def update_bias_terms_via_api(terms, action=None):
    """Update bias terms via the FastAPI backend."""
    try:
        payload = {"terms": terms, "action": action}
        response = requests.post(f"{API_BASE_URL}/bias-terms", json=payload)
        if response.status_code == 200:
            res_json = response.json()
            st.success(res_json.get("message", "Bias terms updated successfully."))
            logger.info("Bias terms updated via API.")
            return res_json.get("terms", [])
        else:
            st.error(f"Failed to update bias terms: {response.status_code}")
            logger.error(f"Bias terms update API error: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error updating bias terms: {e}")
        logger.error(f"Error updating bias terms: {e}")
        return []

def perform_analysis_via_api(text, title, email):
    """Send analysis request to the FastAPI backend."""
    try:
        payload = {
            "text": text,
            "title": title
        }
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json=payload
        )

        if response.status_code == 200:
            analysis = response.json()
            analysis['title'] = title if title else "Untitled Article"
            if email != 'guest':
                save_analysis_to_history(analysis)
            st.session_state['current_analysis'] = analysis  # Save to session state
            return analysis
        else:
            st.error(f"Analysis failed with status code {response.status_code}: {response.text}")
            logger.error(f"Analysis API error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error during analysis request: {e}")
        logger.error(f"Error during analysis request: {e}")
        return None

def display_results(data, is_nested=False):
    """Display analysis results."""
    st.markdown(f"## {data.get('title', 'Untitled Article')}")
    st.markdown(f"**Date:** {data.get('timestamp', 'N/A')}")
    st.markdown(f"**Analyzed by:** {data.get('email', 'guest')}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sentiment = data.get('sentiment', {})
        sentiment_label = sentiment.get('label', 'Neutral')
        sentiment_score = sentiment.get('score', 0.0)
        if 'Negative' in sentiment_label:
            sentiment_color = "#dc3545"  # Red for negative tones
        elif 'Positive' in sentiment_label:
            sentiment_color = "#28a745"  # Green for positive tones
        else:
            sentiment_color = "#6c757d"  # Gray for neutral
        st.markdown(f"**Sentiment:** <span style='color:{sentiment_color};'>{sentiment_label}</span>", unsafe_allow_html=True)
        st.write(f"Score: {sentiment_score:.2f} out of 5")

    with col2:
        bias_found = data.get('bias_terms_found', [])
        bias_count = len(bias_found)
        st.markdown("**Bias Terms**")
        st.write(f"Detected: {bias_count}")

    with col3:
        propaganda_presence = data.get('propaganda', {}).get('contains_propaganda', 'No')
        confidence_label = data.get('propaganda', {}).get('confidence_label', 'Low Confidence')
        st.markdown("**Propaganda**")
        st.write(f"Contains Propaganda: {propaganda_presence} ({confidence_label})")

    with col4:
        final_score = data.get('final_score', 0.0)
        st.markdown("**Final Score**")
        st.write(f"{final_score:.2f} / 100")

    st.markdown("---")

    tabs = st.tabs(["Sentiment Analysis", "Bias Detection", "Propaganda Detection"])

    with tabs[0]:
        st.markdown("### Sentiment Analysis")
        st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}'>{sentiment_label}</span>", unsafe_allow_html=True)
        st.write(f"**Average Sentiment Score:** {sentiment_score:.2f} out of 5")

    with tabs[1]:
        st.markdown("### Bias Detection")
        if bias_found:
            st.write(f"**{len(bias_found)} bias terms detected:**")
            for idx, term in enumerate(bias_found, 1):
                st.write(f"{idx}. {term}")
        else:
            st.write("No biased terms detected.")

    with tabs[2]:
        st.markdown("### Propaganda Detection")
        st.write(f"**Contains Propaganda:** {propaganda_presence} ({confidence_label})")
        avg_confidence = data.get('propaganda', {}).get('avg_confidence_value', 0.0)
        st.write(f"**Average Confidence:** {avg_confidence:.2f}")

        detailed_propaganda = data.get('detailed_propaganda', {})
        if detailed_propaganda:
            st.markdown("**Detailed Propaganda Types:**")
            for label, sentences in detailed_propaganda.items():
                st.markdown(f"**{label}:**")
                for sentence_obj in sentences:
                    sentence = sentence_obj.get('sentence', '')
                    score = sentence_obj.get('confidence_value', 0.0)
                    st.write(f"- {sentence} (Confidence: {score:.2f})")
        else:
            st.write("No detailed propaganda detected.")

    st.markdown("---")

    # Generate a unique key for the download_button
    title = data.get('title', 'analysis')
    timestamp = data.get('timestamp', datetime.datetime.now().isoformat())
    download_key = f"download_{sanitize_key(title)}_{sanitize_key(timestamp)}"

    # Download Button
    analysis_json = json.dumps(data, indent=4)
    st.download_button(
        label="Download Analysis as JSON",
        data=analysis_json,
        file_name=f"{sanitize_key(title)}_analysis.json",
        mime="application/json",
        key=download_key
    )

    if st.session_state.get('logged_in', False) and not is_nested:
        st.success("Analysis completed successfully and saved to history.")

def display_comparative_results(analyses):
    """Display comparative analysis results."""
    st.markdown("## Comparative Analysis Results")
    if not analyses:
        st.info("No analyses to compare.")
        return

    summary = []
    for data in analyses:
        sentiment = data.get('sentiment', {})
        sentiment_label = sentiment.get('label', 'Neutral')
        sentiment_score = sentiment.get('score', 0.0)
        bias_count = len(data.get('bias_terms_found', []))
        propaganda_presence = data.get('propaganda', {}).get('contains_propaganda', 'No')
        final_score = data.get('final_score', 0.0)

        summary.append({
            'Title': data.get('title', 'Untitled'),
            'Sentiment': sentiment_label,
            'Sentiment Score': sentiment_score,
            'Bias Count': bias_count,
            'Propaganda Presence': propaganda_presence,
            'Final Score': final_score,
        })

    df = pd.DataFrame(summary)
    st.dataframe(df)

    for data in analyses:
        st.markdown("---")
        display_results(data, is_nested=True)

# --- User Management UI ---
def register_user_ui():
    """User Registration Interface."""
    st.title("Register")
    st.write("Create a new account to access personalized features.")

    with st.form("registration_form"):
        email = st.text_input("Your Email", key="register_email")
        name = st.text_input("Your Name", key="register_name")
        password = st.text_input("Choose a Password", type='password', key="register_password")
        password_confirm = st.text_input("Confirm Password", type='password', key="register_password_confirm")
        submitted = st.form_submit_button("Register")

        if submitted:
            if not email or not name or not password or not password_confirm:
                st.error("Please fill out all fields.")
                return
            if not is_valid_email(email):
                st.error("Please enter a valid email address.")
                return
            if password != password_confirm:
                st.error("Passwords do not match.")
                return
            if not is_strong_password(password):
                st.error("Password must be at least 8 characters long, include uppercase letters, digits, and special characters.")
                return
            # Make API call to register
            try:
                response = requests.post(
                    f"{API_BASE_URL}/register",
                    json={
                        "email": email,
                        "name": name,
                        "password": password
                    }
                )
                if response.status_code == 200:
                    res_json = response.json()
                    if res_json.get("success"):
                        st.success("Registration successful. You can now log in.")
                        logger.info(f"New user registered: {email}")
                    else:
                        st.error(res_json.get("message", "Registration failed."))
                        logger.error(f"Registration failed for user: {email}")
                else:
                    st.error(f"Registration failed with status code {response.status_code}.")
                    logger.error(f"Registration API error for user: {email}")
            except Exception as e:
                st.error(f"Error during registration: {e}")
                logger.error(f"Exception during registration for user {email}: {e}")

def login_user_ui():
    """User Login Interface."""
    st.title("Login")
    st.write("Access your account to view history and customize settings.")

    with st.form("login_form"):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type='password', key="login_password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if not email or not password:
                st.error("Please enter both email and password.")
                return
            # Make API call to login
            try:
                response = requests.post(
                    f"{API_BASE_URL}/login",
                    json={
                        "email": email,
                        "password": password
                    }
                )
                if response.status_code == 200:
                    res_json = response.json()
                    if res_json.get("success"):
                        st.session_state['logged_in'] = True
                        st.session_state['email'] = email
                        # If your FastAPI returns a token, store it here
                        # st.session_state['token'] = res_json.get('token', '')
                        st.success("Logged in successfully.")
                        st.info("You can now proceed to Single Article Analysis or Comparative Analysis from the sidebar.")
                        logger.info(f"User '{email}' logged in successfully.")
                    else:
                        st.error(res_json.get("message", "Login failed."))
                        logger.warning(f"Failed login attempt for email: '{email}'.")
                else:
                    st.error(f"Login failed with status code {response.status_code}.")
                    logger.error(f"Login API error for user: {email}")
            except Exception as e:
                st.error(f"Error during login: {e}")
                logger.error(f"Exception during login for user {email}: {e}")

def reset_password_flow_ui():
    """Password Reset Interface."""
    st.title("Reset Password")
    st.write("Enter your email and new password.")

    with st.form("reset_password_form"):
        email = st.text_input("Email", key="reset_email")
        new_password = st.text_input("New Password", type='password', key="new_password")
        new_password_confirm = st.text_input("Confirm New Password", type='password', key="new_password_confirm")
        reset_button = st.form_submit_button("Reset Password")

        if reset_button:
            if not email or not new_password or not new_password_confirm:
                st.error("Please fill out all fields.")
                return
            if not is_valid_email(email):
                st.error("Please enter a valid email address.")
                return
            if new_password != new_password_confirm:
                st.error("Passwords do not match.")
                return
            if not is_strong_password(new_password):
                st.error("Password must be at least 8 characters long, include uppercase letters, digits, and special characters.")
                return
            # Make API call to reset password
            try:
                response = requests.post(
                    f"{API_BASE_URL}/reset_password",
                    json={
                        "email": email,
                        "new_password": new_password
                    }
                )
                if response.status_code == 200:
                    res_json = response.json()
                    if res_json.get("success"):
                        st.success("Password reset successful. You can now log in.")
                        logger.info(f"User '{email}' reset their password.")
                    else:
                        st.error(res_json.get("message", "Password reset failed."))
                        logger.error(f"Password reset failed for user: {email}")
                else:
                    st.error(f"Password reset failed with status code {response.status_code}.")
                    logger.error(f"Password reset API error for user: {email}")
            except Exception as e:
                st.error(f"Error during password reset: {e}")
                logger.error(f"Exception during password reset for user {email}: {e}")

def logout_user():
    """Log out the current user."""
    email = st.session_state.get('email', 'guest')
    logger.info(f"User '{email}' logged out.")
    st.session_state['logged_in'] = False
    st.session_state['email'] = ''
    # If using tokens, clear them here
    # st.session_state['token'] = ''
    st.sidebar.success("Logged out successfully.")

# --- Analysis UI Functions ---
def single_article_analysis():
    """Single Article Analysis Interface."""
    st.header("Single Article Analysis")
    st.write("Enter the article URL or paste the article text below.")

    input_type = st.radio(
        "Select Input Type",
        ['Enter URL', 'Paste Article Text'],
        key="single_article_input_type"
    )
    if input_type == 'Enter URL':
        url = st.text_input(
            "Article URL",
            placeholder="https://example.com/article",
            key="single_article_url"
        ).strip()
        article_text = ''
    else:
        article_text = st.text_area(
            "Article Text",
            placeholder="Paste the article text here...",
            height=300,
            key="single_article_text"
        ).strip()
        url = ''

    title = st.text_input(
        "Article Title",
        value="Article",
        placeholder="Enter a title for the article",
        key="single_article_title"
    )

    # Display previous analysis if exists
    if 'current_analysis' in st.session_state and st.session_state['current_analysis']:
        st.markdown("### Previous Analysis Results:")
        display_results(st.session_state['current_analysis'])

    if st.button("Analyze", key="analyze_single_article"):
        if input_type == 'Enter URL':
            if url:
                if is_valid_url(url):
                    with st.spinner('Fetching the article...'):
                        article_text_fetched = fetch_article_text(url)
                        if article_text_fetched:
                            article_text = preprocess_text(article_text_fetched)
                            st.success("Article text fetched successfully.")
                        else:
                            st.error("Failed to fetch article text.")
                            return
                else:
                    st.error("Please enter a valid URL.")
                    return
            else:
                st.error("Please enter a URL.")
                return
        else:
            if not article_text.strip():
                st.error("Please paste the article text.")
                return
            article_text = preprocess_text(article_text)

        with st.spinner('Performing analysis...'):
            email = st.session_state.get('email', 'guest')
            analysis = perform_analysis_via_api(article_text, title, email)
            if analysis:
                st.success("Analysis completed successfully and saved to history.")
                display_results(analysis)
            else:
                st.error("Failed to perform analysis on the provided article.")

def comparative_analysis():
    """Comparative Analysis Interface."""
    st.header("Comparative Analysis")
    st.write("Compare multiple articles side by side.")

    num_articles = st.number_input("Number of articles to compare", min_value=2, max_value=5, value=2, step=1)

    article_texts = []
    titles = []
    analyses = []

    for i in range(int(num_articles)):
        st.subheader(f"Article {i+1}")
        input_type = st.radio(
            f"Select Input Type for Article {i+1}",
            ['Enter URL', 'Paste Article Text'],
            key=f"comp_input_type_{i}"
        )
        if input_type == 'Enter URL':
            url = st.text_input(
                f"Article URL for Article {i+1}",
                placeholder="https://example.com/article",
                key=f"comp_url_{i}"
            ).strip()
            article_text = ''
            if url:
                if is_valid_url(url):
                    with st.spinner(f'Fetching article {i+1}...'):
                        article_text_fetched = fetch_article_text(url)
                        if article_text_fetched:
                            article_text = preprocess_text(article_text_fetched)
                            st.success(f"Article {i+1} text fetched successfully.")
                        else:
                            st.error(f"Failed to fetch article {i+1} text.")
                            return
                else:
                    st.error(f"Please enter a valid URL for Article {i+1}.")
                    return
        else:
            article_text = st.text_area(
                f"Article Text for Article {i+1}",
                height=200,
                key=f"comp_text_{i}"
            )
            if not article_text.strip():
                st.error(f"Please paste the article text for Article {i+1}.")
                return
            article_text = preprocess_text(article_text)

        title = st.text_input(f"Title for Article {i+1}", key=f"comp_title_{i}")
        titles.append(title if title else f"Article {i+1}")
        article_texts.append(article_text)

    if st.button("Analyze Articles", key="compare_articles"):
        if not all(article_texts):
            st.error("Please provide text for all articles.")
            return

        with st.spinner("Performing analysis on all articles..."):
            email = st.session_state.get('email', 'guest')
            for i, text in enumerate(article_texts):
                title = titles[i]
                analysis = perform_analysis_via_api(text, title, email)
                if analysis:
                    analyses.append(analysis)
                else:
                    st.error(f"Failed to analyze Article {i+1}.")
                    return
            if analyses:
                st.success("Comparative Analysis Completed and saved to history.")
                display_comparative_results(analyses)
            else:
                st.error("No analyses to display.")

# --- Help Page Function ---
def help_feature():
    """Help and Documentation Interface."""
    st.header("Help")
    st.write("""
    **Media Bias Detection Tool** helps you analyze articles for sentiment, bias, and propaganda.

    **Single Article Analysis:**  
    - Provide a URL or paste article text.
    - Run analysis to get sentiment, bias terms, and propaganda details.

    **Comparative Analysis:**  
    - Provide multiple articles (via URL or text).
    - Compare sentiment, bias, and propaganda across articles.

    **User Management:**  
    - Register for an account.
    - Log in to save analyses to your history.
    - Reset password if forgotten.

    **History:**  
    - View past analyses if logged in.

    **Download Results:**  
    - After performing an analysis, you can download the results as a JSON file.

    **Note on Sentiment:**  
    The sentiment analysis is based on a pretrained model and may not always match human intuition. "Very Negative" through "Very Positive" provides a gradient for easier interpretation.

    If you encounter any issues or have questions, please contact support.
    """)

# --- History Page Function ---
def display_history():
    """Display Analysis History for Logged-in Users."""
    if not st.session_state.get('logged_in', False):
        st.warning("Please log in to access your history.")
        return

    email = st.session_state.get('email', 'guest')
    history = load_user_history(email)

    st.header("Your Analysis History")
    if not history:
        st.info("No analysis history available.")
        return

    for idx, analysis in enumerate(history[::-1], start=1):  # Display latest first
        with st.expander(f"{analysis.get('title', f'Analysis {idx}')} - {analysis.get('timestamp', 'N/A')}"):
            display_results(analysis, is_nested=True)

# --- Main Function ---
def main():
    """Main function to orchestrate the Streamlit app."""
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'email' not in st.session_state:
        st.session_state['email'] = ''
    if 'current_analysis' not in st.session_state:
        st.session_state['current_analysis'] = None

    # Sidebar Navigation
    st.sidebar.title("Media Bias Detection Tool")
    st.sidebar.markdown("---")
    if not st.session_state['logged_in']:
        menu_items = ["Single Article Analysis", "Comparative Analysis", "Help", "Login", "Register"]
    else:
        menu_items = ["Single Article Analysis", "Comparative Analysis", "History", "Help", "Logout"]

    page = st.sidebar.radio(
        "Navigate to",
        menu_items,
        index=0  # Default to "Single Article Analysis"
    )
    st.sidebar.markdown("---")

    # Display selected page
    if page == "Login":
        if st.session_state['logged_in']:
            st.sidebar.info(f"Already logged in as {st.session_state['email']}.")
            st.sidebar.button("Logout", on_click=logout_user)
        else:
            login_user_ui()
    elif page == "Register":
        if st.session_state['logged_in']:
            st.sidebar.info(f"Already registered as {st.session_state['email']}.")
            st.sidebar.button("Logout", on_click=logout_user)
        else:
            register_user_ui()
    elif page == "Single Article Analysis":
        single_article_analysis()
    elif page == "Comparative Analysis":
        comparative_analysis()
    elif page == "History":
        display_history()
    elif page == "Logout":
        logout_user()
    elif page == "Help":
        help_feature()

    # Show Logout button if logged in and not on Login/Register/History/Logout pages
    if st.session_state['logged_in'] and page not in ["Login", "Register", "History", "Logout"]:
        st.sidebar.markdown(f"**Logged in as:** {st.session_state['email']}")
        if st.sidebar.button("Logout"):
            logout_user()

if __name__ == "__main__":
    main()
