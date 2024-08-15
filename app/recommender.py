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
    sample = model.predict(data_preprocessing(soil_data))
    crops = cluster_data.loc[cluster_data['cluster'] == sample[0], 'label'].values
    return set(crops)



# Example usage
soil_data = [16, 75, 21, 18, 23, 5, 87]  # Example input
recommended_crops = predict_crops(soil_data)
print("Recommended crops in order of preference:", recommended_crops)
