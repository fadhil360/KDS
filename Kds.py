import pandas as pd
import plotly.express as px
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

# Ambil hanya warna dan habitat pertama
df['Primary_Color'] = df['Color'].apply(lambda x: str(x).split(',')[0].strip())
df['Primary_Habitat'] = df['Habitat'].apply(lambda x: str(x).split(',')[0].strip())

# Convert height to numeric
df['Height (cm)'] = df['Height (cm)'].apply(range_to_median)

# Drop missing values
df.dropna(subset=['Height (cm)', 'Primary_Color', 'Primary_Habitat'], inplace=True)

# Encode color
le = LabelEncoder()
df['Color_encoded'] = le.fit_transform(df['Primary_Color'])

# Select features and scale
X = df[['Height (cm)', 'Color_encoded']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Map cluster to most common first habitat
cluster_to_habitat = df.groupby('Cluster')['Primary_Habitat'].agg(lambda x: x.mode()[0])
df['Assigned_Habitat'] = df['Cluster'].map(cluster_to_habitat)

# Interactive Plotly visualization
fig = px.scatter(
    df,
    x='Height (cm)',
    y='Color_encoded',
    color='Cluster',
    hover_data={
        'Primary_Color': True,
        'Primary_Habitat': True,
        'Assigned_Habitat': True,
        'Color_encoded': False  # hide numeric label from hover
    },
    title='K-Means Clustering with Habitat Assignment',
    labels={'Color_encoded': 'Color (Encoded)'}
)
fig.show()

# === Input from user ===
input_height = input("Enter height in cm (e.g., '50-70' or 'up to 60'): ")
input_color = input("Enter color (must match known colors, e.g., 'Grey', 'Brown', etc.): ")

# Process user input
input_height_val = range_to_median(input_height)

if input_color in le.classes_:
    input_color_encoded = le.transform([input_color])[0]
else:
    print(f"Unknown color '{input_color}'. Known colors: {list(le.classes_)}")
    exit()

# Prepare and predict
input_data = pd.DataFrame([[input_height_val, input_color_encoded]], columns=['Height (cm)', 'Color_encoded'])
input_scaled = scaler.transform(input_data)
predicted_cluster = kmeans.predict(input_scaled)[0]
predicted_habitat = cluster_to_habitat[predicted_cluster]

# Output result
print(f"\nPredicted Cluster: {predicted_cluster}")
print(f"Predicted Habitat: {predicted_habitat}")
