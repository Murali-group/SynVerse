import copy
def find_string_match(list1, list2):
    # Given two lists/sets of strings this function will return the common values in the list
    # this will NOT be case sensitive. Also, not sensitive to having/not having '-'/'_'/' '.
    #make all lower case and remove '-','_' and ' '
    list1_dict = {x:x.lower().replace('_','').replace('-','').replace(' ','') for x in list(list1)}
    list2_rename = {x:x.lower().replace('_','').replace('-','').replace(' ','') for x in list(list2)}
    common = set(list1_dict.values()).intersection(set(list2_rename.values()))
    return common, list1_dict, list2_rename

def convert_cell_line_name(name):
    #removing ‘ ’, ‘_’, ‘-’, lowercase
    return name.lower().replace('_', '').replace('-', '').replace(' ', '')

def sort_paired_cols(df, col1, col2, inplace, relation='greater'):
    '''
    Given a pandas dataframe,df and two column names, this function will return a
    dataframe where col1>col2.
    '''
    if relation=='greater':
        mask = df[col1] < df[col2]
    elif relation=='less':
        mask = df[col1] > df[col2]

    if inplace:
        df.loc[mask, [col1, col2]] = df.loc[mask, [col2, col1]].values
    else:
        df_new = copy.deepcopy(df)
        df_new.loc[mask, [col1, col2]] = df_new.loc[mask, [col2, col1]].values
        return df_new


# Function to rename duplicates
def rename_duplicates(series):
    counts = series.value_counts()
    # Creating a list to hold the new names in the order of their appearance
    count_dup = 0
    for gene, count in counts.items():
        if count>1:
            count_dup+=1

    renamed_list = []
    seen = {}
    for item in series:
        if counts[item] > 1:
            if item in seen:
                seen[item] += 1
            else:
                seen[item] = 0
            renamed_list.append(f"{item}_dup_{seen[item]}")
        else:
            renamed_list.append(item)
    print('duplicate genes: ', count_dup)
    print('total unique genes: ', len(set(series)))
    return renamed_list