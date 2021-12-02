import itertools
import copy
def generate_cell_line_idx_mapping(synergy_df):

    cell_lines = synergy_df['Cell_line'].unique()
    cell_line_2_idx = {cell_line: i for i, cell_line in enumerate(cell_lines)}
    idx_2_cell_line = {i: cell_line for i, cell_line in enumerate(cell_lines)}
    return cell_line_2_idx, idx_2_cell_line


def prepare_alg_param_list(alg_name, config_map):
    # reduce_dim = config_map['genex_data_settings']['reduce_dim']
    settings = copy.deepcopy(config_map['ml_models_settings']['algs'][alg_name])
    settings.pop('should_run')
    params = []

    for key in settings:
        params.append(settings[key])

    param_value_combo = itertools.product(*params)
    param_dict_list = [dict(zip(list(settings.keys()), values)) for values in param_value_combo]

    print(param_dict_list)
    return param_dict_list

def prepare_synergy_data_param(config_map):
    synergy_settings = config_map['synergy_data_settings']
    param = ''

    if synergy_settings['apply_range']:
        param += 'range_' + str(synergy_settings['min'])+'_'+ str(synergy_settings['max'])+'_'

    if synergy_settings['apply_threshold']:
        param += 'th_' +  str(synergy_settings['threshold']) + '_'

    if synergy_settings['take_top']:
        param += 'top_'+ str(synergy_settings['n_top_cell_lines']) + '_'+ str( synergy_settings['percent'])+'_'

    if config_map['split']['type']=='test_rare_cell_lines':
        param += str(synergy_settings['n_rare_cell_lines'])+'_'
    param = param[:-1]

    return param


def prepare_split_param(config_map):
    split_settings = config_map['split']
    param = ''

    param += 'neg_'+ split_settings['sampling']+'_' + str(split_settings['neg_frac'])+\
             '_val_'+ str(split_settings['val_frac'])

    return param
def prepare_output_prefix(split_type, config_map, **kwargs):
    out_params =  split_type + '/' + prepare_synergy_data_param(config_map)+'_'+\
                  prepare_split_param(config_map)+ '_'+\
                  kwargs.get('cvdir')
    return out_params


