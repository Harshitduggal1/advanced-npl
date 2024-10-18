import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import TFAutoModel, AutoTokenizer
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Load data from CSV file
def load_data(filepath):
    data = pd.read_csv(filepath)
    sequences = data['sequence']
    labels = data['label']  # Assuming binary labels (0 = no impact, 1 = detrimental)
    return sequences, labels

# Encode protein sequences with a transformer model
def encode_sequences(sequences, model_name='Rostlab/prot_bert_bfd', max_len=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(sequences.tolist(), padding=True, truncation=True, max_length=max_len, return_tensors='tf')
    return inputs

# Create a Transformer-based model
def create_model(model_name='Rostlab/prot_bert_bfd', num_classes=1):
    base_model = TFAutoModel.from_pretrained(model_name)
    input_ids = layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
    
    # Get transformer model outputs
    outputs = base_model(input_ids, attention_mask=attention_mask)
    x = outputs[0][:, 0, :]  # CLS token
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(num_classes, activation='sigmoid')(x)  # Assuming binary classification
    
    model = models.Model(inputs=[input_ids, attention_mask], outputs=x)
    return model

# Main function to run the algorithm
def main(filepath):
    sequences, labels = load_data(filepath)

    # Encode protein sequences
    inputs = encode_sequences(sequences)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    # Split the dataset into training and test sets
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
        input_ids, attention_mask, encoded_labels, test_size=0.2, random_state=42
    )

    # Create the model
    model = create_model(model_name='Rostlab/prot_bert_bfd', num_classes=1)
    model.compile(optimizer=Adam(learning_rate=3e-5), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        [X_train_ids, X_train_mask],
        y_train,
        epochs=5,
        batch_size=32,
        validation_data=([X_test_ids, X_test_mask], y_test)
    )

    # Evaluate the model
    loss, accuracy = model.evaluate([X_test_ids, X_test_mask], y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Example usage
if __name__ == "__main__":
    main('data.csv')  # Replace with your actual dataset file path
