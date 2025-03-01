from rdkit import Chem
from rdkit.Chem import Draw
import sys

def generate_molecule_image(smiles, output_file="molecule.pdf"):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")
        Draw.MolToFile(mol, output_file, size=(100, 100))
        print(f"Molecule image saved as {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    generate_molecule_image(smiles)
