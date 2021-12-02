import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

drug_dir = '/home/tasnina/Projects/SynVerse/datasets/drug-comb-db/'
drug_response_file = drug_dir + 'drugcombs_response.csv'
drug_synergy_file = drug_dir + 'drugcombs_scored.csv'

drug_response_df = pd.read_csv(drug_response_file)[['BlockID','DrugRow', 'DrugCol','ConcRow','ConcCol','Response','source']]
drug_synergy_df = pd.read_csv(drug_synergy_file)
drug_synergy_df = drug_synergy_df[drug_synergy_df['ID'].isin(list(drug_response_df['BlockID']))]

print(drug_response_df.nunique())
print(drug_synergy_df.nunique())
# print(drug_response_df['source'].unique())

#write drug response from different sources in different files
# drug_response_df[drug_response_df['source']=='ONEIL'].to_csv(drug_dir + 'drugcombs_response_ONEIL.csv')
# drug_response_df[drug_response_df['source']=='CLOUD'].to_csv(drug_dir + 'drugcombs_response_CLOUD.csv')
# drug_response_df[drug_response_df['source']=='ALMANAC'].to_csv(drug_dir + 'drugcombs_response_ALMANAC.csv')
# drug_response_df[drug_response_df['source']=='nih'].to_csv(drug_dir + 'drugcombs_response_nih.csv')

drug_1_response_df = drug_response_df[(drug_response_df['ConcCol']==0)][['BlockID','DrugRow','ConcRow','Response']]
drug_2_response_df = drug_response_df[(drug_response_df['ConcRow']==0)][['BlockID','DrugCol','ConcCol','Response']]

drug_1_response = drug_1_response_df.groupby(by = ['BlockID']).mean()
drug_2_response = drug_2_response_df.groupby(by = ['BlockID']).mean()

drug_1_2_single_response = pd.merge(drug_1_response, drug_2_response,\
                                    how='inner', left_index=True, right_index=True)
drug_1_2_single_response['diff'] = abs(drug_1_2_single_response['Response_x']-\
                                   drug_1_2_single_response['Response_y'])

plt.plot(drug_1_response['Response'][0:1000], color='r')
plt.plot(drug_2_response['Response'][0:1000], color='g')
plt.show()


plt.plot(drug_1_2_single_response['diff'][0:9000], color='b', label='activity diff')
plt.plot(drug_synergy_df['Loewe'][0:9000], color = 'r', label = 'Loewe')
# plt.plot(drug_synergy_df['Bliss'][0:1000], color = 'g')
# plt.plot(drug_synergy_df['HSA'][0:1000], color = 'c')
# plt.plot(drug_synergy_df['ZIP'][0:1000], color = 'm')
plt.legend()
plt.show()

l = min(len(drug_1_2_single_response), len(drug_synergy_df))
loewe_corr, _ = pearsonr(drug_1_2_single_response['diff'][0:l], drug_synergy_df['Loewe'][0:l])
print('Loewe corr value: ',loewe_corr)

bliss_corr, _ = pearsonr(drug_1_2_single_response['diff'][0:l], drug_synergy_df['Bliss'][0:l])
print('Bliss corr value: ',bliss_corr)

ZIP_corr, _ = pearsonr(drug_1_2_single_response['diff'][0:l], drug_synergy_df['ZIP'][0:l])
print('ZIP corr value: ',ZIP_corr)

loewe_corr, _ = pearsonr(drug_1_2_single_response['Response_x'][0:l], drug_synergy_df['Loewe'][0:l])
print('Loewe corr value: ',loewe_corr)

loewe_corr, _ = pearsonr(drug_1_2_single_response['Response_y'][0:l], drug_synergy_df['Loewe'][0:l])
print('Loewe corr value: ',loewe_corr)
