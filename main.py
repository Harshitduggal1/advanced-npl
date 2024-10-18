
import os  # 🗂️ Importing os module for file operations
import numpy as np  # 📊 Importing NumPy for numerical operations
import torch  # 🔥 Importing PyTorch for deep learning
from torch.utils.data import DataLoader  # 🚚 Importing DataLoader for batching
import torch.nn as nn  # ⚙️ Importing neural network module from PyTorch
import torch.optim as optim  # ⚡ Importing optimization algorithms
from sklearn.model_selection import train_test_split  # 📚 Importing function to split data

# Function to predict structure
def predict_structure(model, input_sequence, device):  
    input_ids = tokenize(input_sequence).to(device)  # 🧬 Tokenizing the input sequence for the model
    with torch.no_grad():  # 🚫 Disabling gradient computation for inference
        predicted_structure = model(input_ids).cpu().numpy()  # 🧪 Getting predicted structure and moving to CPU
    return predicted_structure  # 📥 Returning the predicted structure

# Function to visualize the predicted structure
def visualize_structure(pdb_file, predicted_structure):  
    cmd.load(pdb_file)  # 📂 Loading the pdb file for visualization
    for i, (resn, resi, x, y, z) in enumerate(predicted_structure):  # 🔄 Iterating through predicted structure
        cmd.alter(f'resn {resn} and resi {resi}', f'x={x}, y={y}, z={z}')  # 🛠️ Altering coordinates of residues
    cmd.show_as('cartoon')  # 🎨 Displaying structure in cartoon representation
    cmd.viewport(800, 600)  # 🖥️ Setting viewport size
    cmd.zoom('all')  # 🔍 Zooming into the structure
    cmd.png('predicted_structure.png')  # 📸 Saving visualization as PNG
    cmd.delete('all')  # 🗑️ Deleting all loaded structures from visualization environment

# Main function execution starts here
if __name__ == '__main__':  
    # Load and preprocess data  
    protein_sequences = []  # 🧪 List to store protein sequences
    protein_structures = []  # 🏗️ List to store protein structures
    for file in os.listdir('data/'):  # 📂 Looping through files in the 'data' directory
        if file.endswith('.fasta'):  # 📄 Checking if file is a FASTA file
            sequence = open(os.path.join('data/', file), 'r').read().splitlines()[1]  # 📝 Reading sequence line
            protein_sequences.append(sequence)  # ➕ Adding sequence to list
            pdb_file = os.path.join('data/', file.replace('.fasta', '.pdb'))  # 🧩 Defining corresponding PDB file path
            protein_structures.append(extract_structural_features(pdb_file))  # 🏗️ Extracting structural features
    protein_sequences = np.array(protein_sequences)  # 🔄 Converting list to NumPy array
    protein_structures = np.array(protein_structures)  # 🔄 Converting list to NumPy array

    # Extract sequence and structural features  
    sequence_features = np.array([extract_sequence_features(seq) for seq in protein_sequences])  # 🌄 Extracting sequence features
    structural_features = protein_structures  # 🏗️ Assigning structural features

    # Split the data into training and validation sets  
    X_train, X_val, y_train, y_val = train_test_split(sequence_features, structural_features, test_size=0.2, random_state=42)  # 📊 Splitting data

    # Create the PyTorch dataset and dataloader  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 💻 Selecting device: GPU if available
    train_dataset = ProteinDataset(X_train, y_train, device)  # 🥗 Creating training dataset
    val_dataset = ProteinDataset(X_val, y_val, device)  # 🥗 Creating validation dataset
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 🚚 Creating data loader for training
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 🚚 Creating data loader for validation

    # Define the model, loss function, and optimizer  
    model = ProteinStructureModel(input_size=len(extract_sequence_features('DUMMY')), hidden_size=512, output_size=15, device=device).to(device)  # 🏗️ Initializing the model
    criterion = nn.MSELoss()  # 📉 Defining loss function (Mean Squared Error)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # ⚡ Initializing Adam optimizer

    # Train the model  
    num_epochs = 100  # 🔁 Setting number of epochs
    train_model(model, train_dataloader, criterion, optimizer, num_epochs)  # 🏋️ Training the model

    # Evaluate the model on the validation set  
    val_loss = evaluate_model(model, val_dataloader, criterion)  # 📉 Evaluating validation loss
    print(f'Validation Loss: {val_loss}')  # 📣 Printing validation loss

    # Predict and visualize the structure for a new sequence  
    new_sequence = 'MKTVRQERLKSIVRILERSKEPNPQPGTTHDIVSRWALSTYLNGADFMPVLGFGTYAYPGKITFNEHGRQSIRGNMKDKPVRGAQLANGALRMIPASQWVIRNVSEEVAQLKKNIGIALIKTNNCALADALLDSVPTPSNPPTTEEEKTESNQPEVTCVVVTDSQYALNGNGNEVTMTLHFMFLRLNARGRKTLRSIAFPQTDINLFLNGSLVDGQTGPHKIQGINALCAIHPQYLAKANASWRIFLQSDRYKYWDVNEV'  # 🧬 Defining new protein sequence
    predicted_structure = predict_structure(model, new_sequence, device)  # 🔮 Predicting structure for the new sequence
    visualize_structure('data/new_sequence.pdb', predicted_structure)  # 🎨 Visualizing the predicted structure
