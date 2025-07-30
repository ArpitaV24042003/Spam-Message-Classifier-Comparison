import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
    return df

df = load_data()

# --- Preprocessing ---
encoder = LabelEncoder()
df['label_encoded'] = encoder.fit_transform(df['label'])

# Add spammy phrase detection as a binary feature
def spammy_phrases(message):
    spam_words = ['congratulations', 'verify', 'account', 'lottery', 'free', 'urgent',
                  'click here', 'winner', 'selected', 'limited time', 'act now']
    message_lower = message.lower()
    return any(word in message_lower for word in spam_words)

df['has_spammy_words'] = df['message'].apply(spammy_phrases).astype(int)

X = df['message']
y = df['label_encoded']

# Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_raw)
X_test_tfidf = vectorizer.transform(X_test_raw)

# Combine TF-IDF with spammy word feature
X_train_feat = hstack([
    X_train_tfidf,
    df.loc[X_train_raw.index, 'has_spammy_words'].values.reshape(-1, 1)
])
X_test_feat = hstack([
    X_test_tfidf,
    df.loc[X_test_raw.index, 'has_spammy_words'].values.reshape(-1, 1)
])

# Balance training data with SMOTE
smote = SMOTE()
X_train_bal, y_train_bal = smote.fit_resample(X_train_feat, y_train)

# --- Train Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_bal, y_train_bal)
y_pred_rf = rf_model.predict(X_test_feat)

# --- Train Naive Bayes (no SMOTE, just TF-IDF) ---
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

# --- Evaluation ---
def evaluate_model(y_true, y_pred, model_name):
    return pd.Series({
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }, name=model_name)

results = pd.concat([
    evaluate_model(y_test, y_pred_rf, "Random Forest"),
    evaluate_model(y_test, y_pred_nb, "Naive Bayes")
], axis=1)

# --- Streamlit UI ---
st.title("Spam Message Classifier Comparison")

# Show model comparison
st.subheader("Model Performance Comparison")
st.dataframe(results.T.style.background_gradient(cmap='Blues'))

# --- Prediction Functions ---
def predict_spam_rf(email_text, threshold=0.4):
    email_tfidf = vectorizer.transform([email_text])
    spam_feature = int(spammy_phrases(email_text))
    email_combined = hstack([email_tfidf, [[spam_feature]]])
    
    prediction_proba = rf_model.predict_proba(email_combined)[0]
    spam_prob = prediction_proba[encoder.transform(['spam'])[0]]
    ham_prob = prediction_proba[encoder.transform(['ham'])[0]]
    
    prediction_label = 'spam' if spam_prob > threshold else 'ham'
    
    return {
        'model': 'Random Forest',
        'label': prediction_label,
        'ham_prob': ham_prob,
        'spam_prob': spam_prob
    }

def predict_spam_nb(email_text, threshold=0.5):
    email_tfidf = vectorizer.transform([email_text])
    
    prediction_proba = nb_model.predict_proba(email_tfidf)[0]
    spam_prob = prediction_proba[encoder.transform(['spam'])[0]]
    ham_prob = prediction_proba[encoder.transform(['ham'])[0]]
    
    prediction_label = 'spam' if spam_prob > threshold else 'ham'
    
    return {
        'model': 'Naive Bayes',
        'label': prediction_label,
        'ham_prob': ham_prob,
        'spam_prob': spam_prob
    }

# --- Predict Custom Email ---
st.subheader("Try Your Own Message")
user_input = st.text_area("Enter a message to classify using both models:", "")

if st.button("Predict"):
    if user_input.strip():
        st.write(f"**Email:** {user_input}")
        
        # Predict using both models
        rf_result = predict_spam_rf(user_input)
        nb_result = predict_spam_nb(user_input)

        # Display Random Forest result
        st.markdown("### 🌲 Random Forest Prediction")
        st.write(f"**Predicted Label:** {rf_result['label'].upper()}")
        st.write("**Confidence:**")
        st.write(f"- Ham: {rf_result['ham_prob']:.2%}")
        st.write(f"- Spam: {rf_result['spam_prob']:.2%}")

        # Display Naive Bayes result
        st.markdown("### 📘 Naive Bayes Prediction")
        st.write(f"**Predicted Label:** {nb_result['label'].upper()}")
        st.write("**Confidence:**")
        st.write(f"- Ham: {nb_result['ham_prob']:.2%}")
        st.write(f"- Spam: {nb_result['spam_prob']:.2%}")
    else:
        st.warning("Please enter a message to classify.")

# --- Notes ---
st.subheader("Notes")
st.markdown("""
- TF-IDF features are enhanced with a binary spam phrase detector
- SMOTE used for class balancing (Random Forest only)
- Naive Bayes still performs well but lacks extra feature
- Lowering the spam threshold helps flag borderline spam
""")
