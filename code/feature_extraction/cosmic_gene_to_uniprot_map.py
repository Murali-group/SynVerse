import pandas as pd

dir = '/home/tasnina/Projects/SynVerse/inputs/cell_lines/models_07_13_can_use/'
gene_expression_file_name = dir+ 'uncompressed_gene_expression.tsv'
gene_name_file = dir +'cosmic_genes.tsv'
gene_expression_df = pd.read_csv(gene_expression_file_name, sep='\t')

gene_names = list(gene_expression_df.columns)
print(len(gene_names))

gene_names = pd.Series(gene_names)
gene_names.to_csv(gene_name_file, index=False)

