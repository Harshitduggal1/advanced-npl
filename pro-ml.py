import os  
import numpy as np  
import pandas as pd  
import torch  
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader  
from sklearn.model_selection import train_test_split  
from transformers import BertModel, BertTokenizer  
from Bio.Seq import Seq  
from Bio.Alphabet import generic_protein  
from Bio.SeqUtils.ProtParam import ProteinAnalysis  
from pymol import cmd, stored  

# Data Preprocessing  
class ProteinDataset(Dataset):  
    def __init__(self, sequences, structures):  
        self.sequences = sequences  
        self.structures = structures  
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  

    def __len__(self):  
        return len(self.sequences)  

    def __getitem__(self, idx):  
        sequence = self.sequences[idx]  
        structure = self.structures[idx]  
        input_ids = self.tokenizer.encode(sequence, padding='max_length', max_length=512, truncation=True)  
        return torch.tensor(input_ids), torch.tensor(structure)  

# Feature Engineering  
def extract_sequence_features(sequence):  
    seq = Seq(sequence, generic_protein)  
    analyzer = ProteinAnalysis(str(seq))  
    return [  
        analyzer.molecular_weight(),  
        analyzer.aromaticity(),  
        analyzer.instability_index(),  
        analyzer.isoelectric_point(),  
        analyzer.gravy()  
    ]  

def extract_structural_features(pdb_file):  
    # Load the PDB file and extract structural features  
    cmd.load(pdb_file)  
    stored.residues = []  
    cmd.iterate("all", "stored.residues.append((resn, resi, x, y, z))")  
    cmd.delete("all")  
    return stored.residues  

# Model Architecture  
class ProteinStructureModel(nn.Module):  
    def __init__(self, input_size, hidden_size, output_size):  
        super(ProteinStructureModel, self).__init__()  
        self.bert = BertModel.from_pretrained('bert-base-uncased')  
        self.linear1 = nn.Linear(input_size, hidden_size)  
        self.dropout = nn.Dropout(0.2)  
        self.linear2 = nn.Linear(hidden_size, output_size)  

    def forward(self, input_ids):  
        bert_output = self.bert(input_ids)[0][:, 0, :]  
        x = self.linear1(bert_output)  
        x = self.dropout(x)  
        x = self.linear2(x)  
        return x  

# Training and Evaluation  
def train_model(model, dataloader, criterion, optimizer, device, num_epochs):  
    model.train()  
    for epoch in range(num_epochs):  
        running_loss = 0.0  
        for inputs, targets in dataloader:  
            inputs, targets = inputs.to(device), targets.to(device)  
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, targets)  
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')  

    print('Finished Training')  

def evaluate_model(model, dataloader, device):  
    model.eval()  
    total_loss = 0  
    criterion = nn.MSELoss()  
    with torch.no_grad():  
        for inputs, targets in dataloader:  
            inputs, targets = inputs.to(device), targets.to(device)  
            outputs = model(inputs)  
            loss = criterion(outputs, targets)  
            total_loss += loss.item()  
    return total_loss / len(dataloader)  

# Prediction and Visualization  
def predict_structure(model, sequence, device):  
    model.eval()  
    input_ids = torch.tensor([model.tokenizer.encode(sequence, padding='max_length', max_length=512, truncation=True)]).to(device)  
    with torch.no_grad():  
        predicted_structure = model(input_ids).cpu().numpy()  
    return predicted_structure  

def visualize_structure(pdb_file, predicted_structure):  
    cmd.load(pdb_file)  
    for i, (resn, resi, x, y, z) in enumerate(predicted_structure):  
        cmd.alter(f'resn {resn} and resi {resi}', f'x={x}, y={y}, z={z}')  
    cmd.show_as('cartoon')  
    cmd.viewport(800, 600)  
    cmd.zoom('all')  
    cmd.png('predicted_structure.png')  
    cmd.delete('all')  

# Main  
if __name__ == '__main__':  
    # Load and preprocess data  
    protein_sequences = []  
    protein_structures = []  
    for file in os.listdir('data/'):  
        if file.endswith('.fasta'):  
            sequence = open(os.path.join('data/', file), 'r').read().splitlines()[1]  
            protein_sequences.append(sequence)  
            pdb_file = os.path.join('data/', file.replace('.fasta', '.pdb'))  
            protein_structures.append(extract_structural_features(pdb_file))  
    protein_sequences = np.array(protein_sequences)  
    protein_structures = np.array(protein_structures)  

    # Extract sequence and structural features  
    sequence_features = np.array([extract_sequence_features(seq) for seq in protein_sequences])  
    structural_features = protein_structures  

    # Split the data into training and validation sets  
    X_train, X_val, y_train, y_val = train_test_split(sequence_features, structural_features, test_size=0.2, random_state=42)  

    # Create the PyTorch dataset and dataloader  
    train_dataset = ProteinDataset(X_train, y_train)  
    val_dataset = ProteinDataset(X_val, y_val)  
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)  

    # Set up the device (CPU or GPU)  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

    # Define the model, loss function, and optimizer  
    model = ProteinStructureModel(input_size=5, hidden_size=256, output_size=15).to(device)  
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  

    # T# Define the optimizer and learning rate scheduler
initial_lr = 0.001  # Use a smaller learning rate
optimizer = Adam(model.parameters(), lr=initial_lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

# Update the training loop to include validation loss checking
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train_loss = train_model(model, train_dataloader, criterion, optimizer, device)
    val_loss = evaluate_model(model, val_dataloader, device)
    
    print(f'Train Loss: {train_loss}, Validation Loss: {val_loss}')

    # Step the scheduler
    scheduler.step(val_loss)

    # Save model checkpoint if validation loss improved
    if val_loss < best_val_loss:  # best_val_loss should be defined and stored initially
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# Load best model for prediction
model.load_state_dict(torch.load('best_model.pth'))

# Predict and visualize the structure for a new sequence
new_sequence = 'MKTVRQERLKSIVRILERSKEPNPQPGTTHDIVSRWALSTYLNGADFMPVLGFGTYAYPGKITFNEHGRQSIRGNMKDKPVRGAQLANGALRMIPASQWVIRNVSEEVAQLKKNIGIALIKTNNCALADALLDSVPTPSNPPTTEEEKTESNQPEVTCVVVTDSQYALNGNGNEVTMTLHFMFLRLNARGRKTLRSIAFPQTDINLFLNGSLVDGQTGPHKIQGINALCAIHPQYLAKANASWRIFLQSDRYKYWDVNEV'  
predicted_structure = predict_structure(model, new_sequence, device)  
visualize_structure('data/new_sequence.pdb', predicted_structure)
