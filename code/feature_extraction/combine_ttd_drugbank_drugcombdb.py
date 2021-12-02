import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(os.getcwd())
os.chdir('../../')
print(os.getcwd())

dataset_dir = "datasets/"
processed_drug_feature_path = dataset_dir + 'processed/drug/'
drug_pair_score_file = dataset_dir+ 'drug-comb-db/' + 'drugcombs_scored.csv'

drugcombdb_drug_chem_smiles_maccskeys_info_file =  processed_drug_feature_path + 'drug_chem_smiles_maccskeys_info_file.tsv'
drugbank_drug_target_mapping_file = processed_drug_feature_path+ 'drug_target_map_drugbank.tsv'
TTD_drug_target_mapping_file= processed_drug_feature_path+'drug_target_map_TTD.tsv'
drugcombdb_drug_target_links_file = dataset_dir+ 'drug-comb-db/' + 'drug_protein_links.tsv'
STRINGID_uniprot_map_file = dataset_dir + 'mappings/human/'+ 'STRINGID_uniprot_map.tsv'

input_synergy_dir = "inputs/synergy/"
input_drug_dir = "inputs/drugs/"
TTD_drugbank_drug_target_map_file = input_drug_dir+'drug_target_map_only_ttd_drug_bank.tsv'
single_drug_training_maccskeys_target_feature_file_path = input_drug_dir + 'single_drugs_maccskeys_target_features.tsv'
single_drug_training_maccskeys_feature_file_path = input_drug_dir + 'single_drugs_maccskeys_features.tsv'
training_label_synergy_score_file_path =  input_synergy_dir + 'synergy_labels.tsv'
final_drug_target_map_file = 'inputs/drugs/drug_target_map.tsv' 
output_dir = "outputs/"


#*************************** convert MACCSKEYS to 166 columns ************************************

drug_maccskeys_df = pd.read_csv(drugcombdb_drug_chem_smiles_maccskeys_info_file, sep = '\t')
drug_maccskeys_df = drug_maccskeys_df[['pubchem_cid','Drug_Name','MACCSKeys','MACCSKeys_bitstring']].rename(columns={'Drug_Name':'drug_name'})

drug_maccskeys_string_dict = {}
for index, row in drug_maccskeys_df.iterrows():
    drug_maccskeys_string_dict[row['pubchem_cid']] = list(row['MACCSKeys_bitstring'])
drug_maccskeys_df_feature_ready = pd.DataFrame.from_dict(drug_maccskeys_string_dict, orient = 'index').                                reset_index().rename(columns={'index':'pubchem_cid'})
drug_maccskeys_df_feature_ready['pubchem_cid'] = drug_maccskeys_df_feature_ready['pubchem_cid'].astype(str)
print(drug_maccskeys_df_feature_ready.shape)
# print(drug_maccskeys_df_feature_ready.head())

# print(drug_maccskeys_df_feature_ready['pubchem_cid'].unique())
#drop column as it does not contain any meaningful info
drug_maccskeys_df_feature_ready.drop([0],axis=1,inplace=True)
print(drug_maccskeys_df_feature_ready.shape)
# print(drug_maccskeys_df_feature_ready.head())

os.makedirs(os.path.dirname(single_drug_training_maccskeys_feature_file_path), exist_ok=True)
drug_maccskeys_df_feature_ready.to_csv(single_drug_training_maccskeys_feature_file_path, sep = '\t',index=False)



#*********************** READ SYNERGY FILE *********************************
synergy_df = pd.read_csv(drug_pair_score_file).rename(columns={'Cell line': 'Cell_line'})
synergy_df = synergy_df.groupby(['Drug1', 'Drug2', 'Cell_line']). agg({'Loewe': 'mean', 'Bliss': 'mean', 'ZIP': 'mean'}).reset_index()
# drugs1 = set(synergy_df['Drug1']).union(set(synergy_df['Drug2']))
# print(len(drugs1))

# print(synergy_df.head())
drugcombdb_drugname_pubchem_cid_map = drug_maccskeys_df[['drug_name','pubchem_cid']].set_index('drug_name')
# print(drugcombdb_drugname_pubchem_cid_map.head())

synergy_df = synergy_df[(synergy_df['Drug1'].isin(drugcombdb_drugname_pubchem_cid_map.index)) & \
                        (synergy_df['Drug2'].isin(drugcombdb_drugname_pubchem_cid_map.index))]
# print(synergy_df.head())

synergy_df['Drug1_pubchem_cid'] = synergy_df['Drug1'].apply(lambda x:drugcombdb_drugname_pubchem_cid_map.at[x,'pubchem_cid'] ).astype(str)
synergy_df['Drug2_pubchem_cid'] = synergy_df['Drug2'].apply(lambda x:drugcombdb_drugname_pubchem_cid_map.at[x,'pubchem_cid'] ).astype(str)

drugs = set(synergy_df['Drug1_pubchem_cid']).union(set(synergy_df['Drug2_pubchem_cid']))

print(len(drugs))
#********************************* Combine protein target data from TTD, Drugbank and MACCSKEYS data for drugs in drugcombdb *************

#combine drugbank and TTD drug-target information

drugbank_drug_target_df = pd.read_csv(drugbank_drug_target_mapping_file, sep='\t')
drugbank_drug_target_df['pubchem_cid'] = drugbank_drug_target_df['pubchem_cid'].astype(str)

TTD_drug_target_df = pd.read_csv(TTD_drug_target_mapping_file, sep='\t')
TTD_drug_target_df['pubchem_cid'] = TTD_drug_target_df['pubchem_cid'].astype(str)

drug_target_df = pd.concat([drugbank_drug_target_df, TTD_drug_target_df], axis= 0)


print('drugbank:', drugbank_drug_target_df.head())
print('TTD: ', TTD_drug_target_df.head())

os.makedirs(os.path.dirname(TTD_drugbank_drug_target_map_file), exist_ok=True)

df = drug_target_df[['pubchem_cid','uniprot_id']].drop_duplicates()
print(df.head())
df.to_csv(TTD_drugbank_drug_target_map_file, sep='\t')
print(len(drug_target_df))
#note: 1. a pubchem id can be mapped to multiple drug_name
#     2. a drug_name can be mapped to multiple pubchem_id
#     So many to many relationship


# In[5]:


#combine TTD and DrugBank target info with DrugCombDB target info
#read the STRING_ID to uniprot_id mapping file
string_to_uniprot_df = pd.read_csv(STRINGID_uniprot_map_file, sep='\t')
print(string_to_uniprot_df.head())
string_to_uniprot_dict = dict(zip(string_to_uniprot_df['STRING'], string_to_uniprot_df['uniprot']))

#now read the drug_comb_db drug_target info file. Rename column 'chemical' as 'pubchem_id'
drug_target_df = pd.read_csv(drugcombdb_drug_target_links_file, sep='\t')
drug_target_df.head()
drug_target_df['pubchem_cid'] = drug_target_df['chemical'].apply(lambda x: x.replace('CIDm',''))

#filter out the target proteins from drug_comb_db that do not have uniprot id mapping
drug_target_df= drug_target_df[drug_target_df['protein'].isin(string_to_uniprot_dict.keys())]
drug_target_df['uniprot_id'] = drug_target_df['protein'].apply(lambda x:string_to_uniprot_dict[x])                           
print(drug_target_df.head())

#only keep columns 'pubchem_cid' and 'uniprot_id'
drug_target_df = drug_target_df[['pubchem_cid','uniprot_id']]

#load drug target map containing targets from TTD and drugbank and combine it with target info from drugcombdb
ttd_drug_bank_drug_target_map_df = pd.read_csv(TTD_drugbank_drug_target_map_file, sep='\t')
final_drug_target_df = pd.concat([ttd_drug_bank_drug_target_map_df,drug_target_df], axis=0).reset_index(drop=True)
drug_target_df['pubchem_cid'] = drug_target_df['pubchem_cid'].apply(lambda x: x.replace('CIDs',''))
final_drug_target_df.drop_duplicates(inplace=True)


# In[6]:


#TTD DrugBank DrugCombDB all target info combined in one file
final_drug_target_df = final_drug_target_df[['pubchem_cid','uniprot_id']]
#keep only the drugs present in synergy file
final_drug_target_df = final_drug_target_df[final_drug_target_df['pubchem_cid'].isin(drugs)]
final_drug_target_df.to_csv(final_drug_target_map_file, sep='\t')
print(final_drug_target_df.head())
print(len(final_drug_target_df))


# In[7]:


#create the feature matrix containing all the targets for a drug as a 0-1 vector.
final_drug_target_df =  pd.read_csv(final_drug_target_map_file, sep='\t', dtype=str)
unique_proteins = final_drug_target_df['uniprot_id'].unique()
unique_proteins_in_Index_form = pd.Index(pd.Series(unique_proteins))
number_of_unique_proteins = len(unique_proteins)
print('number of unique proteins: ', number_of_unique_proteins)
unique_pubchem_ids = final_drug_target_df['pubchem_cid'].unique()
print('number of drugs: ',len(unique_pubchem_ids))
drug_targets ={ pubchem_id: np.zeros(number_of_unique_proteins, dtype=int) for pubchem_id in  unique_pubchem_ids}

for index,row in final_drug_target_df.iterrows():
    #find the index of the current target (i.e.present in the current row) in unique_proteins
    prot_position = unique_proteins_in_Index_form.get_loc(row['uniprot_id'])
    drug_targets[row['pubchem_cid']][prot_position] = 1

drug_target_df_feature_ready = pd.DataFrame.from_dict(drug_targets, orient='index', columns=unique_proteins)
drug_target_df_feature_ready = drug_target_df_feature_ready.reset_index().rename(columns={'index':'pubchem_cid'})
drug_target_df_feature_ready['pubchem_cid']=drug_target_df_feature_ready['pubchem_cid'].astype(str)
print(drug_target_df_feature_ready.head())
print(drug_target_df_feature_ready.shape)
print(drug_target_df_feature_ready['pubchem_cid'].unique())





#this is the final feature matrix containing pubchem_id : targets as 0-1 vector : Maccskeys
#after this concatenation only the drugs having both target ans maccs keys info remain
drug_related_training_feature_df = drug_maccskeys_df_feature_ready.set_index('pubchem_cid').\
    join(drug_target_df_feature_ready.set_index('pubchem_cid'), how='inner').reset_index()
# if (not os.path.exists(input_train_dir)):
#     print('hello')
#     os.makedirs(input_train_dir)
#
os.makedirs(os.path.dirname(single_drug_training_maccskeys_target_feature_file_path), exist_ok=True)
drug_related_training_feature_df.to_csv(single_drug_training_maccskeys_target_feature_file_path,index=False, sep = '\t')


# In[ ]:


drug_related_training_feature_df = pd.read_csv(single_drug_training_maccskeys_target_feature_file_path, sep='\t', index_col=None)
drug_related_training_feature_df['pubchem_cid'] = drug_related_training_feature_df['pubchem_cid'].astype(str)
print(drug_related_training_feature_df.head())
print(drug_related_training_feature_df.shape)


# # Create training labels: Training ready synergy score values for drugpairs

# In[ ]:


# create a training ready file containing drug1_pubchem_id, drug_2_pubchem_id, cell_line, loewe score


#keep only the synergy score for pairs of drugs for which we have other drug features i.e. maccs keys and target available
synergy_df = synergy_df[(synergy_df['Drug1_pubchem_cid'].isin(drug_related_training_feature_df['pubchem_cid'])) & \
                        (synergy_df['Drug2_pubchem_cid'].isin(drug_related_training_feature_df['pubchem_cid']))]
#remove pairs where drug1==drug2
synergy_df = synergy_df[synergy_df['Drug1_pubchem_cid']!= synergy_df['Drug2_pubchem_cid']]

print(synergy_df.head())


# In[ ]:


def std(x): return np.std(x)


# In[ ]:


synergy_df = synergy_df[['Drug1_pubchem_cid','Drug2_pubchem_cid','Cell_line','Loewe','Bliss','ZIP']]
# 

#if there is multiple entry for same drug-pairs with slightly different names in druncombdb scored file,
#then this will cause multiple entry of a single drugpair for particular cell line.
#I will take the average of those entries

# print(type(synergy_df['Drug2_pubchem_cid'].values[0]))
synergy_df['max_drug'] = synergy_df[['Drug1_pubchem_cid','Drug2_pubchem_cid']].astype(int).max(axis=1).astype(str)
synergy_df['min_drug'] = synergy_df[['Drug1_pubchem_cid','Drug2_pubchem_cid']].astype(int).min(axis=1).astype(str)

#from multiple source we can get same drug-drug-cell_line tuples with different Loewe score. In such cases,
#keep the drug-drug-cellline tuples for which score from multiple sources do not differ by more than 10.

df = synergy_df.groupby(['max_drug','min_drug','Cell_line']).                                        agg({'Loewe':[std,'mean','min','max'],'Bliss': 'mean', 'ZIP':'mean'}).reset_index()
df.columns = ['_'.join(col).strip() for col in df.columns.values]
# print(df.head())
print(synergy_df.shape)
print(df.shape)



# In[ ]:


df['std_by_mean'] = df['Loewe_std']/df['Loewe_mean'].abs()
synergy_df = df[df['std_by_mean']<=0.02]

synergy_df['Drug1_pubchem_cid'] = synergy_df['max_drug_']
synergy_df['Drug2_pubchem_cid'] = synergy_df['min_drug_']
synergy_df['Cell_line'] = synergy_df['Cell_line_']
synergy_df['Loewe'] = synergy_df['Loewe_mean']
synergy_df['Bliss'] = synergy_df['Bliss_mean']
synergy_df['ZIP'] = synergy_df['ZIP_mean']

synergy_df = synergy_df[['Drug1_pubchem_cid', 'Drug2_pubchem_cid','Cell_line','Loewe','Bliss','ZIP']].                reset_index(drop=True)
print(len(synergy_df), synergy_df.head(10))

#here the max drug is in synergy_df[Drug1_pubchem_cid] and min drug in 'Drug2_pubchem_cid'.\
#But I do not want to put any such restriction. so swapping drug_1 and drug_2 randomly.
synergy_df_swap = synergy_df.copy()
synergy_df_swap['Drug1_pubchem_cid'] = synergy_df['Drug2_pubchem_cid']
synergy_df_swap['Drug2_pubchem_cid'] = synergy_df['Drug1_pubchem_cid']
print(len(synergy_df_swap), synergy_df_swap.head(10))


#take some pairs from synergy_df and other from synergy_df_swap
np.random.seed(1)
is_original = np.random.choice([True,False], size=len(synergy_df))

# concat to make new dataset
synergy_df = pd.concat((synergy_df[is_original],synergy_df_swap[~is_original])).reindex(range(len(synergy_df)))
print(len(synergy_df),synergy_df.head(10))

os.makedirs(os.path.dirname(training_label_synergy_score_file_path), exist_ok=True)
synergy_df.to_csv(training_label_synergy_score_file_path, sep='\t', index = False)


# In[ ]:


data =np.abs(list(df['std_by_mean']))
# print(data)
# print(df['Loewe_std'])
print(len(data))
plt.figure(figsize=(6, 4))
x = plt.hist(data,  weights=np.zeros_like(data) + 1. / len(data), bins=10, range=(0,0.2))
# x = plt.hist(data,range=(0,50),bins=5)
plt.xlabel('std/mean of Loewe score')
plt.ylabel('fraction of total tuples')

plt.savefig(output_dir+'/Viz/standard_deviation_Loewe_score.png')
plt.savefig(output_dir+'/Viz/standard_deviation_Loewe_score.pdf')
plt.show()
print(x)


# # statistical analysis

# In[ ]:


#Make sure that you are preserving the datatype while reading it from file
synergy_df = pd.read_csv(training_label_synergy_score_file_path, sep='\t',                          dtype = {'Drug1_pubchem_cid': str ,'Drug2_pubchem_cid': str,                         'Cell_line':str, 'Loewe': np.float64, 'Bliss': np.float64, 'ZIP': np.float64},                         index_col = None)

print(synergy_df.head())
print(synergy_df.columns)

print(synergy_df['Drug1_pubchem_cid'].unique())


# In[ ]:


#single pubchem to multiple drug_name mapping stat in drugbank
drugbank_drug_target_grouped_df = drugbank_drug_target_df.groupby('pubchem_cid')['drug_name'].unique().reset_index()
drugbank_drug_target_grouped_df['#drug_name_per_pubchem_cid'] = drugbank_drug_target_grouped_df['drug_name'].                                                            apply(lambda x: len(x))

drugbank_drug_target_df_multiple_drugname_per_pubchem_id = drugbank_drug_target_grouped_df[drugbank_drug_target_grouped_df['#drug_name_per_pubchem_cid']>1]
print(drugbank_drug_target_df_multiple_drugname_per_pubchem_id)


# In[ ]:


#single pubchem to multiple drug_name mapping stat: TTD
TTD_drug_target_grouped_df = TTD_drug_target_df.groupby('pubchem_cid')['drug_name'].unique().reset_index()
TTD_drug_target_grouped_df['#drug_name_per_pubchem_cid'] = TTD_drug_target_grouped_df['drug_name'].                                                            apply(lambda x: len(x))

TTD_drug_target_multiple_drugname_per_pubchem_id_df = TTD_drug_target_grouped_df[TTD_drug_target_grouped_df['#drug_name_per_pubchem_cid']>1]
print(TTD_drug_target_multiple_drugname_per_pubchem_id_df)
print('total pubchem id with multiple drug name:', len(TTD_drug_target_multiple_drugname_per_pubchem_id_df))


# In[ ]:


#single drug_name to multiple pubchem mapping stat
drugbank_drug_target_grouped_df = drugbank_drug_target_df.groupby('drug_name')['pubchem_cid'].unique().reset_index()
drugbank_drug_target_grouped_df['#pubchem_cid_per_drug'] = drugbank_drug_target_grouped_df['pubchem_cid'].                                                            apply(lambda x: len(x))

drugbank_drug_target_df_multiple_pubchem_id_per_drugname = drugbank_drug_target_grouped_df[drugbank_drug_target_grouped_df['#pubchem_cid_per_drug']>1]
print(drugbank_drug_target_df_multiple_pubchem_id_per_drugname)


# In[ ]:


#single drug_name to multiple pubchem mapping stat
TTD_drug_target_grouped_df = TTD_drug_target_df.groupby('drug_name')['pubchem_cid'].unique().reset_index()
TTD_drug_target_grouped_df['#pubchem_cid_per_drug'] = TTD_drug_target_grouped_df['pubchem_cid'].                                                            apply(lambda x: len(x))

TTD_drug_target_df_multiple_pubchem_id_per_drugname = TTD_drug_target_grouped_df[TTD_drug_target_grouped_df['#pubchem_cid_per_drug']>1]
print(TTD_drug_target_df_multiple_pubchem_id_per_drugname)


# In[ ]:


#find out for how many drugcombdb drug pubchem ids we have target information
drugs_in_drugcombdb = set(drug_maccskeys_df_feature_ready['pubchem_cid'].astype(str))
drugs_having_target_info = set(drug_target_df_feature_ready['pubchem_cid'].astype(str))

print('total drugcombdb drug: ', len(drugs_in_drugcombdb))
print('target info present for: ', len(drugs_having_target_info))
print('common: ',len(drugs_in_drugcombdb.intersection(drugs_having_target_info)))

# print(drugs_in_drugcombdb, '\n')
# print(drugs_having_target_info)

common = drugs_in_drugcombdb.intersection(drugs_having_target_info)


# In[ ]:


#find out for how many drugcombdb drug-pairs we have target information for both drugs
cell_lines  = synergy_df['Cell_line'].unique()
drug_pairs_per_cell_line_having_all_feature_val={x:0 for x in cell_lines}

# synergy_df = synergy_df.reset_index()


for row in synergy_df.itertuples():
        drug_pairs_per_cell_line_having_all_feature_val[row.Cell_line] += 1

print(drug_pairs_per_cell_line_having_all_feature_val) 


# In[ ]:


#plot_hist
number_of_drug_pairs_list = list((drug_pairs_per_cell_line_having_all_feature_val.values()))
number_of_drug_pairs_list.sort()
print(number_of_drug_pairs_list)

bins_seq = list(np.arange(0,3000,100))
# bins_seq = bins_seq+[20000,30000,40000,50000]

# print(bins_seq)

fig, ax = plt.subplots()
ax.hist(drug_pairs_per_cell_line_having_all_feature_val.values(),bins=bins_seq,alpha=0.5,label='#total drug pairs')
# y_label = 'number of cell lines (' + str(len(number_of_drug_pairs_list)) + ')'
y_label = 'number of cell lines'
x_label = 'number of drug pairs'

ax.set_ylabel(y_label)
ax.set_xlabel(x_label)
# ax.legend()

# plt.title('#drug combinations per cell line for which all feature values are available')
# file_name = output_dir+'/Viz/'+ 'drug_pairs_per_cell_line.pdf'
# os.makedirs(os.path.dirname(file_name), exist_ok=True)
# plt.savefig(file_name, format= 'pdf', bbox_inches = 'tight' )
plt.show()


# In[ ]:


count =0
for x in number_of_drug_pairs_list:
    if x>2000:
        count+=1
print (count)


# In[ ]:




