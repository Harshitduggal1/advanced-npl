import os  
import numpy as np  
import pandas as pd  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  
from sklearn.model_selection import train_test_split  
from transformers import BertModel, BertTokenizer, AlbertModel, AlbertTokenizer  
from Bio.Seq import Seq  
from Bio.Alphabet import generic_protein  
from Bio.SeqUtils.ProtParam import ProteinAnalysis  
from pymol import cmd, stored  
from rdkit import Chem  
from rdkit.Chem import AllChem  
from rdkit.Chem.Descriptors import *  
from scipy.spatial.distance import cdist  

# Data Preprocessing  
class ProteinDataset(Dataset):  
    def __init__(self, sequences, structures, device):  
        self.sequences = sequences  
        self.structures = structures  
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')  
        self.device = device  

    def __len__(self):  
        return len(self.sequences)  

    def __getitem__(self, idx):  
        sequence = self.sequences[idx]  
        structure = self.structures[idx]  
        input_ids = self.tokenizer.encode(sequence, padding='max_length', max_length=512, truncation=True)  
        return torch.tensor(input_ids, device=self.device), torch.tensor(structure, device=self.device)  

# Feature Engineering  
def extract_sequence_features(sequence):  
    seq = Seq(sequence, generic_protein)  
    analyzer = ProteinAnalysis(str(seq))  
    mol = Chem.AddHs(Chem.MolFromSequence(sequence))  
    AllChem.EmbedMolecule(mol)  
    conformer = mol.GetConformer()  
    coords = np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])  
    return [  
        analyzer.molecular_weight(),  
        analyzer.aromaticity(),  
        analyzer.instability_index(),  
        analyzer.isoelectric_point(),  
        analyzer.gravy(),  
        *list(GetMolecularFormulaAndWeight(mol).values()),  
        *list(GetMolWt(mol), GetNumHeavyAtoms(mol), GetNumRotatableBonds(mol), GetNumHDonors(mol), GetNumHAcceptors(mol)),  
        *list(cdist(coords, coords).flatten())  
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
    def __init__(self, input_size, hidden_size, output_size, device):  
        super(ProteinStructureModel, self).__init__()  
        self.albert = AlbertModel.from_pretrained('albert-base-v2')  
        self.linear1 = nn.Linear(input_size, hidden_size)  
        self.dropout = nn.Dropout(0.2)  
        self.linear2 = nn.Linear(hidden_size, output_size)  
        self.device = device  

    def forward(self, input_ids):  
        albert_output = self.albert(input_ids)[0][:, 0, :]  
        x = self.linear1(albert_output)  
        x = self.dropout(x)  
        x = self.linear2(x)  
        return x  

# Training and Evaluation  
def train_model(model, dataloader, criterion, optimizer, num_epochs):  
    model.train()  
    for epoch in range(num_epochs):  
        running_loss = 0.0  
        for inputs, targets in dataloader:  
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, targets)  
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')  

    print('Finished Training')  

def evaluate_model(model, dataloader, criterion):  
    model.eval()  
    total_loss = 0  
    with torch.no_grad():  
        for inputs, targets in dataloader:  
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    train_dataset = ProteinDataset(X_train, y_train, device)  
    val_dataset = ProteinDataset(X_val, y_val, device)  
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)  

    # Define the model, loss function, and optimizer  
    model = ProteinStructureModel(input_size=len(extract_sequence_features('DUMMY')), hidden_size=512, output_size=15, device=device).to(device)  
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  

    # Train the model  
    num_epochs = 100  
    train_model(model, train_dataloader, criterion, optimizer, num_epochs)  

    # Evaluate the model on the validation set  
    val_loss = evaluate_model(model, val_dataloader, criterion)  
    print(f'Validation Loss: {val_loss}')  

    # Predict and visualize the structure for a new sequence  
    new_sequence = 'MKTVRQERLKSIVRILERSKEPNPQPGTTHDIVSRWALSTYLNGADFMPVLGFGTYAYPGKITFNEHGRQSIRGNMKDKPVRGAQLANGALRMIPASQWVIRNVSEEVAQLKKNIGIALIKTNNCALADALLDSVPTPSNPPTTEEEKTESNQPEVTCVVVTDSQYALNGNGNEVTMTLHFMFLRLNARGRKTLRSIAFPQTDINLFLNGSLVDGQTGPHKIQGINALCAIHPQYLAKANASWRIFLQSDRYKYWDVNEV'  
    predicted_structure = predict_structure(model, new_sequence, device)  
    visualize_structure('data/new_sequence.pdb', predicted_structure)
