#***************** CELL LINE Preprocess ***************************
import pandas as pd
def landmark_gene_filter(genex_df, lincs_file):
    landmark_genes_df = pd.read_csv(lincs_file,sep='\t', skiprows=lambda x: x == 1)[['L1000 Probe ID','Gene Entrez ID', 'Gene Symbol']]
    #remove the control/invariant genes/probes
    landmark_genes_df = landmark_genes_df[~landmark_genes_df['L1000 Probe ID'].str.contains('INV', case=True)]

    landmark_genes = list(set(landmark_genes_df['Gene Symbol']))

    filtered_genex_df = pd.concat([genex_df['cell_line_name'],genex_df[[col for col in genex_df.columns if col in landmark_genes]]], axis=1)
    print(len(filtered_genex_df.columns))
    print(len(genex_df.columns))
    return filtered_genex_df
