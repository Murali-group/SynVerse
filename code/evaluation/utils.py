from dataclasses import dataclass
@dataclass
class EvalScore:
    auprc: float
    auroc: float
    early_prec: float

def set_title_suffix(model_param):
    title_suffix = ''
    for key in model_param:
        title_suffix = title_suffix +'_' + key +'_' + str(model_param[key])
    return title_suffix

def keep_one_from_symmetric_pairs(df, aggregate = 'max'):
    # df will contain predicted score for all (x,y) and (y,x) pairs from a single run
    # df =  df_input
    df = sort_max_drug_first(df, 'drug_1','drug_2')
    # print('df after sorting\n', df.head())
    if (aggregate == 'max'):
        df = df.groupby(['drug_1','drug_2','cell_line','true'], as_index = False)['predicted'].max()
        # print('df after taking max:\n', df.head())
    elif(aggregate=='mean'):
        df = df.groupby(['drug_1', 'drug_2', 'cell_line', 'true'], as_index=False)['predicted'].mean()
    return df


def sort_max_drug_first(df, drug1_col, drug2_col):
    #this function will take a dataframe df as input
    #sort df such that in sorted df, max(drug1_col, drug2_col) will be in drug1_col and min(drug1_col, drug2_col)\
    # will be in drug2_col
    # df = df_input
    df['max_drug'] = df[[drug1_col, drug2_col]].max(axis=1)
    df['min_drug'] = df[[drug1_col, drug2_col]].min(axis=1)
    # print(df.head(10))
    df[drug1_col] = df['max_drug']
    df[drug2_col] = df['min_drug']
    df.drop(['max_drug', 'min_drug'], axis=1, inplace=True)
    # print(df.head(10))
    return df
