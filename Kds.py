import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Helper function to compute median from a string range
def range_to_median(value):
    try:
        value = str(value).lower().strip()
        if value.startswith("up to"):
            high = float(value.replace("up to", "").strip())
            return high / 2
        elif '-' in value:
            parts = value.split('-')
            low, high = float(parts[0]), float(parts[1])
            return (low + high) / 2
        else:
            return float(value)
    except:
        return None

# Load dataset
df = pd.read_csv('Animal Dataset.csv')
# Ambil hanya warna pertama dari kolom 'Color'
df['Primary_Color'] = df['Color'].apply(lambda x: str(x).split(',')[0].strip())
# Convert and encode
df['Height (cm)'] = df['Height (cm)'].apply(range_to_median)
le = LabelEncoder()
df['Color_encoded'] = le.fit_transform(df['Primary_Color'])

# Drop missing values
df.dropna(subset=['Height (cm)', 'Primary_Color', 'Habitat'], inplace=True)

# Extract only the first habitat
df['Primary_Habitat'] = df['Habitat'].apply(lambda x: str(x).split(',')[0].strip())

# Select features
X = df[['Height (cm)', 'Color_encoded']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Map cluster to most common first-listed habitat
cluster_to_habitat = df.groupby('Cluster')['Primary_Habitat'].agg(lambda x: x.mode()[0])
df['Assigned_Habitat'] = df['Cluster'].map(cluster_to_habitat)

# Visualize
plt.scatter(X['Height (cm)'], X['Color_encoded'], c=df['Cluster'], cmap='viridis')
plt.title('K-Means Clustering with Habitat Assignment')
plt.xlabel('Height (cm)')
plt.ylabel('Color (Encoded)')
plt.show()

# Optional: Show mapping result
print(df[['Height (cm)', 'Color', 'Primary_Habitat', 'Cluster', 'Assigned_Habitat']].head())
# === Input from user ===
input_height = input("Enter height in cm (can be a range or number, e.g., '50-70' or 'up to 60'): ")
input_color = input("Enter color (must match known colors, e.g., 'Grey', 'Brown', etc.): ")

# Process height input
input_height_val = range_to_median(input_height)

# Encode input color (bandingkan dengan kelas dari Primary_Color)
if input_color in le.classes_:
    input_color_encoded = le.transform([input_color])[0]
else:
    print(f"Unknown color '{input_color}'. Known colors: {list(le.classes_)}")
    exit()
# Prepare and scale input
input_data = pd.DataFrame([[input_height_val, input_color_encoded]], columns=['Height (cm)', 'Color_encoded'])
input_scaled = scaler.transform(input_data)

# Predict cluster and habitat
predicted_cluster = kmeans.predict(input_scaled)[0]
predicted_habitat = cluster_to_habitat[predicted_cluster]

print(f"\nPredicted Cluster: {predicted_cluster}")
print(f"Predicted Habitat: {predicted_habitat}")
