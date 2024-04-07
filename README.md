##Data Preprocessing

>Load the container event dataset from a JSON file.
>Preprocess the dataset by cleaning the external status descriptions and encoding the internal status labels.

##Model Architecture
>The model architecture consists of a neural network with multiple dense layers.
>The input layer accepts vectorized external status descriptions.
>Hidden layers perform feature transformations.
.The output layer predicts the internal status labels using softmax activation.

##Training Procedure
>Split the preprocessed dataset into training and testing sets.
>Vectorize text data using TensorFlow's TextVectorization layer.
>Standardize features using StandardScaler.
>Compile the model with appropriate optimizer and loss function.
>Train the model on the training set for multiple epochs with mini-batch gradient descent.
>Evaluate the model on the testing set using accuracy, precision, and recall metrics.

##API Implementation
>Develop an API using FastAPI framework.
>Define a request body model to accept external status descriptions as input.
>Implement an endpoint to make predictions based on the provided external status description.
>Load the trained model and necessary preprocessing components within the API.

##Testing Results
>The trained model achieved a certain level of accuracy, precision, and recall on the testing set.
>The API endpoint successfully makes predictions based on the provided external status descriptions.
