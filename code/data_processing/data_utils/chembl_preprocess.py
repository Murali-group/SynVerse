# import math
# from pathlib import Path
# from zipfile import ZipFile
# from tempfile import TemporaryDirectory
#
# import numpy as np
# import pandas as pd
# from rdkit.Chem import PandasTools
# from chembl_webresource_client.new_client import new_client
# # from tqdm.auto import tqdm
#
#
# def main():
#     targets_api = new_client.target
#     compounds_api = new_client.molecule
#     bioactivities_api = new_client.activity


import requests


def find_target_from_drug_name(drug_name):
    # Define the base URL for the CHEMBL API
    base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule"

    # Define the query parameters to search for the given drug name
    params = {
        'molecule_synonyms__molecule_synonym': drug_name
    }

    # Make a GET request to the CHEMBL API
    response = requests.get(base_url, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract the response content as text
        content = response.text

        # Extract target information from the content
        targets = []
        molecules = content.split('"molecules":')[1].split('],"page_meta":')[0]
        for molecule in molecules.split('{"molecule":'):
            if '"molecule_targets":' in molecule:
                targets.extend([target.split('"target_chembl_id":"')[1].split('"')[0]
                                for target in molecule.split('"molecule_targets":')[1].split(']}')[0].split(',') if
                                target.startswith('{"target_chembl_id":"')])

        return targets
    else:
        # Print an error message if the request fails
        print(f"Error: Failed to retrieve data from CHEMBL API. Status code: {response.status_code}")
        return None


# Example usage:
drug_name = "Aspirin"
targets = find_target_from_drug_name(drug_name)
if targets:
    print(f"Targets associated with {drug_name}: {targets}")
else:
    print(f"No targets found for {drug_name}")
