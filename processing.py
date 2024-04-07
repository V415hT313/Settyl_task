import json
import re
import nltk
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import requests
from fastapi import FastAPI
from pydantic import BaseModel

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

url = "https://gist.github.com/farhaan-settyl/ecf9c1e7ab7374f18e4400b7a3d2a161"
response = requests.get(url)
print(response.text)


# Load JSON data
if response.status_code == 200:
    
    with open('cleaned_data.json', 'r') as f:
        cleaned_data = json.loads(response.text)

    # Initialize NLTK resources
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Preprocess each entry
    preprocessed_data = []
    for entry in cleaned_data:

        if 'external_status' not in entry:
            continue

        external_status = entry['external_status']
        internal_status = entry['internal_status']

        # Clean external status
        external_status = re.sub(r'[^\w\s]', '', external_status)  # Remove punctuation
        external_status = external_status.lower()  # Convert to lowercase
        external_status_tokens = word_tokenize(external_status)  # Tokenize
        external_status_tokens = [lemmatizer.lemmatize(token) for token in external_status_tokens if token not in stop_words]  # Lemmatize and remove stop words

        # Clean internal status (assuming it's already in a suitable format)

        preprocessed_data.append({
            'external_status': ' '.join(external_status_tokens),
            'internal_status': internal_status
        })
    # Extract features and labels
    X = np.array([entry['external_status'] for entry in preprocessed_data])
    y = np.array([entry['internal_status'] for entry in preprocessed_data])

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.33, random_state=42)

    if X_train.size > 0:
        print("Number of samples in train set:", len(X_train))
        # Continue with model training and evaluation
        
    else:
        print("Failed to split the data into train and test sets.")
    
    print("Number of samples in train set:", len(X_train))
    print("Number of samples in test set:", len(X_test))
    

    # Vectorize text data using TF-IDF or other methods
    # Here, let's use TensorFlow's TextVectorization layer
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=1000, output_mode='tf-idf')
    vectorizer.adapt(X_train)

    X_train = vectorizer(np.array(X_train)).numpy()
    X_test = vectorizer(np.array(X_test)).numpy()

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define neural network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # Save preprocessed data to JSON file
    with open('cleaned_data.json', 'w') as f:
        json.dump(cleaned_data, f, indent=4)

    print("Number of entries in cleaned_data:", len(cleaned_data))

    # Define FastAPI app
    app = FastAPI()

    # Define request body model
    class Item(BaseModel):
        external_status: str

    # Define endpoint to make predictions
    @app.post("/predict")
    def predict(item: Item):
        external_status = [item.external_status]

        # Vectorize text data
        external_status_vectorized = vectorizer(external_status).numpy()

        # Scale features
        external_status_scaled = scaler.transform(external_status_vectorized)

        # Make prediction
        prediction = model.predict(external_status_scaled)

        # Decode prediction
        predicted_label_index = np.argmax(prediction)

        # Load label encoder
        with open('label_encoder.json', 'r') as f:
            label_encoder_data = json.load(f)
            label_encoder_classes = label_encoder_data['classes']

        predicted_label = label_encoder_classes[predicted_label_index]

        return {"predicted_internal_status": predicted_label}

else:
    print("Failed to fetch data from the JSON file. Status code:", response.status_code)
