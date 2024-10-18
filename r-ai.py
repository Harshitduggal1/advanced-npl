import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from Bio.PDB import PDBParser
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

class ProteinGCN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(ProteinGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

def encode_amino_acids(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    encoding = np.zeros((len(sequence), len(amino_acids)))
    for i, aa in enumerate(sequence):
        if aa in amino_acids:
            encoding[i, amino_acids.index(aa)] = 1
    return encoding

def create_contact_map(pdb_file, threshold=8.0):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    atoms = [atom for atom in structure.get_atoms() if atom.name == 'CA']
    coords = np.array([atom.coord for atom in atoms])
    distances = pdist(coords)
    contact_map = squareform(distances) < threshold
    return contact_map.astype(int)

def prepare_protein_data(sequence, pdb_file):
    features = encode_amino_acids(sequence)
    contact_map = create_contact_map(pdb_file)
    edge_index = torch.tensor(np.array(np.where(contact_map == 1)), dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def load_protein_dataset(fasta_file, pdb_dir):
    dataset = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        pdb_file = f"{pdb_dir}/{record.id}.pdb"
        data = prepare_protein_data(sequence, pdb_file)
        dataset.append(data)
    return dataset

def train_model(model, train_loader, val_loader, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                val_loss += criterion(out, data.y).item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

    return model

def predict_protein_structure(model, protein_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        protein_data = protein_data.to(device)
        prediction = model(protein_data.x, protein_data.edge_index, protein_data.batch)
    return prediction

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    train_dataset = load_protein_dataset("train_proteins.fasta", "train_pdb_dir")
    val_dataset = load_protein_dataset("val_proteins.fasta", "val_pdb_dir")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize and train the model
    num_features = train_dataset[0].num_features
    hidden_channels = 64
    num_classes = 3  # Example: predicting secondary structure (helix, sheet, coil)
    
    model = ProteinGCN(num_features, hidden_channels, num_classes)
    trained_model = train_model(model, train_loader, val_loader)

    # Example prediction
    test_protein = prepare_protein_data("MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADVLASHGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQIRLVLASHGKLQLPAVNRRLKEQVTGSMAKRQPAHTRSGGLSPTSAS", "test_protein.pdb")
    prediction = predict_protein_structure(trained_model, test_protein)
    print("Protein Structure Prediction:", prediction)
