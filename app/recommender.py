import pickle

# Load the trained model
with open('models/RandomForestClassifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/cluster_data.pkl', 'rb') as f:
    cluster_data = pickle.load(f)

def data_preprocessing(data):
    return scaler.transform([data])


def predict_crops(soil_data):
    """Predict crops and order them based on cluster probabilities."""
    # Get probability distribution across clusters
    probabilities = model.predict_proba(data_preprocessing(soil_data))[0]
    
    # Predict the most likely cluster for the given soil data
    predicted_cluster = model.predict(data_preprocessing(soil_data))[0]
    
    # Fetch crops corresponding to the predicted cluster
    crops = cluster_data.loc[cluster_data['cluster'] == predicted_cluster, 'label'].values
    
    # Remove duplicates while preserving order
    unique_crops = list(dict.fromkeys(crops))
    
    # Sort the unique crops based on the probability of their cluster
    ordered_crops = sorted(unique_crops, key=lambda crop: probabilities[predicted_cluster], reverse=True)
    
    # Return the list of unique crops from the predicted cluster, ordered by cluster probability
    return ordered_crops



# Example usage
soil_data = [16, 75, 21, 18, 23, 5, 87]  # Example input
recommended_crops = predict_crops(soil_data)
print("Recommended crops in order of preference:", recommended_crops)
