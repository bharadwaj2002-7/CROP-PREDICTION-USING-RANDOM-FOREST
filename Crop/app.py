import numpy as np
import pandas as pd

df=pd.read_csv('Crop_recommendation.csv')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

c=df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target']=c.cat.codes
y=df.target
X=df[['N','P','K','temperature','humidity','ph','rainfall']]

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=4,n_estimators=100,random_state=42).fit(X_train,y_train)
print('RF Accuracy on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('RF Accuracy on test set: {:.2f}'.format(clf.score(X_test, y_test)))
from pyngrok import ngrok
# Streamlit app title and description
import streamlit as st
import pickle
# Load the saved model
#with open('random_forest_model.pkl', 'rb') as model_file:
#    clf = pickle.load(model_file)
# Create a dictionary to map class numbers to crop names
crop_mapping = {
    1: 'rice',
    2: 'maize',
    3: 'chickpea',
    4: 'kidneybeans',
    5: 'pigeonpeas',
    6: 'mothbeans',
    7: 'mungbean',
    8: 'blackgram',
    9: 'lentil',
    10: 'pomegranate',
    11: 'banana',
    12: 'mango',
    13: 'grapes',
    14: 'watermelon',
    15: 'muskmelon',
    16: 'apple',
    17: 'orange',
    18: 'papaya',
    19: 'coconut',
    20: 'cotton',
    21: 'jute',
    22: 'coffee',
    # Add more crops as needed
}
# Streamlit app title and description
st.title("Crop Recommendation App")
st.write("This app uses a Random Forest Classifier to recommend crops based on input features.")
# Create input fields for user to enter feature values
N = st.slider("Nitrogen (N)", min_value=0, max_value=200)
P = st.slider("Phosphorus (P)", min_value=0, max_value=200)
K = st.slider("Potassium (K)", min_value=0, max_value=200)
temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=40.0)
humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0)
ph = st.slider("pH", min_value=0.0, max_value=14.0)
rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=500.0)
# Button to make a prediction
if st.button("Predict Crop"):
    # Prepare the user inputs as a feature vector
    user_inputs = [N, P, K, temperature, humidity, ph, rainfall]
    # Make a prediction using the loaded model
    prediction_class = clf.predict([user_inputs])[0]
    # Get the crop name from the mapping
    predicted_crop = crop_mapping.get(prediction_class, "Unknown Crop")
    st.subheader("Recommended Crop:")
    st.subheader(predicted_crop)

