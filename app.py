import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hybrid Crop Recommender", page_icon="🌾", layout="wide")

# --- STYLE ---
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #00ffcc;
    text-align: center;
}
.stButton>button {
    background-color: #00ffcc;
    color: black;
    border-radius: 10px;
    height: 50px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# --- LOTTIE ---
def load_lottie(url):
    return requests.get(url).json()

lottie = load_lottie("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")
st_lottie(lottie, height=200)

st.title("🌾 Hybrid Crop Recommendation System")
st.markdown("Fuzzy Logic + Explainable AI + Machine Learning 🚀")

# --- DATA ---
@st.cache_data
def load_data():
    return pd.read_csv('Crop_recommendation.csv')

data = load_data()
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# --- FUZZY ---
@st.cache_data
def load_profiles():
    crop_profiles = {}
    for crop in data['label'].unique():
        crop_data = data[data['label'] == crop]
        profile = {}
        for feature in features:
            profile[feature] = {
                'min': crop_data[feature].min(),
                'mean': crop_data[feature].mean(),
                'max': crop_data[feature].max()
            }
        crop_profiles[crop] = profile
    return crop_profiles

crop_profiles = load_profiles()

def triangular_membership(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b-a != 0 else 1.0
    elif b < x < c:
        return (c - x) / (c - b) if c-b != 0 else 1.0
    return 0.0

def fuzzy_recommend(user_input):
    crop_scores = {}
    explanations = {}

    for crop, profile in crop_profiles.items():
        total_score = 0
        contributions = {}

        for f in features:
            val = user_input[f]
            a = profile[f]['min']
            b = profile[f]['mean']
            c = profile[f]['max']

            score = triangular_membership(val, a, b, c)
            contributions[f] = score
            total_score += score

        crop_scores[crop] = (total_score / len(features)) * 100
        explanations[crop] = contributions

    sorted_res = sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_res, explanations

# --- ML ---
@st.cache_data
def load_ml_models():
    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    trained = {}
    acc = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        trained[name] = model
        acc[name] = accuracy_score(y_test, pred)

    return trained, acc

ml_models, ml_acc = load_ml_models()

# --- INPUT ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    n_val = st.slider("Nitrogen", 0, 140, 90)
    p_val = st.slider("Phosphorus", 5, 145, 42)
    k_val = st.slider("Potassium", 5, 205, 43)
    ph_val = st.slider("pH", 3.5, 10.0, 6.5)

with col2:
    temp_val = st.slider("Temperature", 8.0, 45.0, 24.0)
    hum_val = st.slider("Humidity", 14.0, 100.0, 82.0)
    rain_val = st.slider("Rainfall", 20.0, 300.0, 205.0)

# --- BUTTON ---
if st.button("🚀 Recommend Crop"):

    user_input = {
        'N': n_val, 'P': p_val, 'K': k_val,
        'temperature': temp_val, 'humidity': hum_val,
        'ph': ph_val, 'rainfall': rain_val
    }

    results, explanations = fuzzy_recommend(user_input)
    top_crop = results[0][0]

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🌾 Fuzzy", "🤖 ML", "📊 Graphs"])

    # --- FUZZY TAB ---
    with tab1:
        st.metric("🏆 Best Crop", top_crop, f"{results[0][1]:.2f}%")

        st.subheader("🔍 Explanation")

        for f, score in explanations[top_crop].items():

            if score > 0.7:
                level = "highly suitable ✅"
            elif score > 0.4:
                level = "moderately suitable ⚠️"
            else:
                level = "less suitable ❌"

            st.write(f"🔹 {f.capitalize()} is **{level}** ({score:.2f}) for {top_crop}.")
            st.progress(score)

    # --- ML TAB ---
    with tab2:
        for name, model in ml_models.items():
            pred = model.predict(pd.DataFrame([user_input]))[0]
            st.write(f"{name}: **{pred}** (Accuracy: {ml_acc[name]*100:.2f}%)")

    # --- GRAPH TAB ---
    with tab3:
        # Feature graph
        fig1 = px.bar(
            x=list(explanations[top_crop].keys()),
            y=list(explanations[top_crop].values()),
            title="Feature Contribution",
            text=[f"{v:.2f}" for v in explanations[top_crop].values()]
        )
        fig1.update_traces(textposition='outside')
        fig1.update_layout(yaxis_range=[0,1])

        st.plotly_chart(fig1)

        # Comparison graph
        names = list(ml_acc.keys()) + ["Fuzzy"]
        scores = list(ml_acc.values()) + [results[0][1]/100]

        fig2 = px.bar(
            x=names,
            y=scores,
            title="Model Comparison",
            text=[f"{s:.2f}" for s in scores]
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(yaxis_range=[0,1])

        st.plotly_chart(fig2)