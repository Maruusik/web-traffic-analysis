import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Step 1: Data Collection (Assuming data is already collected and stored in a CSV file)
log_file = 'web_server_logs.csv'

# Step 2: Data Preprocessing
# Read the log file into a DataFrame
log_data = pd.read_csv(log_file)

# Drop irrelevant columns or clean data as needed
log_data.drop(columns=['irrelevant_column'], inplace=True)

# Remove duplicates
log_data.drop_duplicates(inplace=True)

# Step 3: Traffic Analysis
# Example: Count requests per client IP
requests_per_ip = log_data['client_ip'].value_counts()

# Step 4: Machine Learning for Anomaly Detection (Using Isolation Forest as an example)
# Preprocess data for anomaly detection
X = log_data[['request_time']].values

# Train Isolation Forest model
model = IsolationForest(contamination=0.1)  # Adjust contamination based on expected anomaly rate
model.fit(X)

# Predict anomalies
anomaly_labels = model.predict(X)
anomalies = log_data[anomaly_labels == -1]  # Extract rows classified as anomalies

# Step 5: Traffic Profiling
# Example: Categorize traffic based on request URLs
url_categories = log_data.groupby('request_url').size()

# Step 6: Visualization
# Example: Visualize requests per client IP
plt.figure(figsize=(10, 6))
requests_per_ip.plot(kind='bar')
plt.title('Requests per Client IP')
plt.xlabel('Client IP')
plt.ylabel('Number of Requests')
plt.show()

# Step 7: Security Threat Detection
# Example: Implement basic rules-based threat detection

# Step 8: Continuous Improvement and Monitoring
# Monitor system performance and update analysis techniques as needed
