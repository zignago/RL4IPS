import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    # Load the dataset (CICIDS2017 dataset)
    dataset = pd.read_csv(file_path)
    
    # Remove leading spaces from column names
    dataset.columns = dataset.columns.str.strip()

    # Feature selection
    features = dataset[['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                        'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 
                        'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean']]
    
    # Label selection (malicious or benign traffic)
    labels = dataset['Label']

    # Feature normalization
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Convert labels to binary format (benign = 0, malicious = 1)
    labels = labels.apply(lambda x: 1 if 'malicious' in x.lower() else 0)

    # Combine features and labels to form the dataset
    data = list(zip(features, labels))

    return data
