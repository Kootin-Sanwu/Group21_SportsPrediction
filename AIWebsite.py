import streamlit as st
import pickle
import numpy as np

try:
    # Load the pre-trained model
    with open("better_random_forest.pkl", "rb") as better_random_forest_file:
        better_random_forest = pickle.load(better_random_forest_file)

    # Load the pre-trained scaler
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    st.error(f"Error loading models: {str(e)}")

# The header of the website
st.title("FIFA Player Prediction")

# The default values of the attributes
default_value = 50
default_value_eur = 2902288
default_wage_eur = 9148
default_age = 25
default_release_clause = 4296353

# Creating a slider for each attribute
potential = st.slider("potential", min_value=0,
                      max_value=100, value=92)

value_eur = st.slider("value_eur", min_value=0,
                      max_value=185500000, value=63000000)

wage_eur = st.slider("wage_eur", min_value=0,
                     max_value=560000, value=220000)

age = st.slider("age", min_value=0, max_value=53, value=35)

release_clause_eur = st.slider(
    "release_clause_eur", min_value=0, max_value=203100000,
    value=75900000)

shooting = st.slider("shooting", min_value=0,
                     max_value=100, value=93)

passing = st.slider("passing", min_value=0,
                    max_value=100, value=81)

dribbling = st.slider("dribbling", min_value=0,
                      max_value=100, value=89)

physic = st.slider("physic", min_value=0,
                   max_value=100, value=77)

attacking_short_passing = st.slider(
    "attacking_short_passing", min_value=0, max_value=100, value=82)

skill_long_passing = st.slider(
    "skill_long_passing", min_value=0, max_value=100, value=77)

movement_reactions = st.slider(
    "movement_reactions", min_value=0, max_value=100, value=95)

power_shot_power = st.slider(
    "power_shot_power", min_value=0, max_value=100, value=94)

mentality_vision = st.slider(
    "mentality_vision", min_value=0, max_value=100, value=82)

mentality_composure = st.slider(
    "mentality_composure", min_value=0, max_value=100, value=95)

mentality_positioning = st.slider(
    "mentality_positioning", min_value=0, max_value=100, value=95)

feature_names = [potential, value_eur, wage_eur, age,
                 release_clause_eur, shooting, passing, dribbling,
                 physic, attacking_short_passing, skill_long_passing,
                 movement_reactions, power_shot_power, mentality_vision,
                 mentality_composure, mentality_positioning]

# Scaling the user inputs
scaled_data = scaler.transform([feature_names])

# Creating a prediction button for making predictions
if st.button("Predict"):
    prediction = better_random_forest.predict(scaled_data)
    if prediction is not None:
        rounded_prediction = round(prediction[0])
        st.write(f"Prediction: {rounded_prediction}")

# Calculating the confidence level
    num_samples = 1000
    predictions = []
    for _ in range(num_samples):
        y_pred = better_random_forest.predict(scaled_data)
        predictions.append(y_pred[0])
    lower_bound = np.percentile(predictions, 2.5)
    upper_bound = np.percentile(predictions, 97.5)

    rounded_lower_bound = round(lower_bound)
    rounded_upper_bound = round(upper_bound)

    st.write(f"Lower Bound: {rounded_lower_bound}")
    st.write(f"Upper Bound: {rounded_upper_bound}")
