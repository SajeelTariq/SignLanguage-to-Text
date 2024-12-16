import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Function to load data from a pickle file
def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    data = np.asarray(data_dict['data'], dtype=object)
    labels = np.asarray(data_dict['labels'])
    
    return data, labels

# Function to train and save the model
def train_and_save_model(data, labels, model_filename):
    # Confirm all data entries are the same length
    data_lengths = [len(item) for item in data]
    if len(set(data_lengths)) > 1:
        raise ValueError("Inconsistent data length. Ensure all feature vectors have the same length.")

    # If data lengths are consistent, convert to a numeric array
    data = np.vstack(data)  # Convert to a 2D array if all elements have the same length

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)

    print(f'{score * 100:.2f}% of samples were classified correctly!')

    # Save the trained model to a file
    with open(model_filename, 'wb') as f:
        pickle.dump({'model': model}, f)

# Load data from both single_hand.pickle and multi_hand.pickle
single_data, single_labels = load_data_from_pickle('./single_hand.pickle')
multi_data, multi_labels = load_data_from_pickle('./multi_hand.pickle')

# Train and save models for single-hand and multi-hand data
train_and_save_model(single_data, single_labels, 'single.p')
train_and_save_model(multi_data, multi_labels, 'multi.p')
