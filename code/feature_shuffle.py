import numpy as np
import random
def shuffle_features(feat_dict):
    '''
    Shuffles the features in the feature dictionary. If the values are numpy arrays, rows are shuffled.
    If the values are dictionaries, the order of the molecular graph data is shuffled.

    :param feat_dict: dict, where:
        - key = feature name
        - value = either a numpy array (each row represents feature values of a drug/cell line)
                  or a dict (key = drug_idx, value = molecular graph data)

    :return: dict with shuffled features
    '''
    shuffled_feat_dict = {}

    for feat_name, feat_value in feat_dict.items():
        if isinstance(feat_value, np.ndarray):
            # Shuffle rows of the numpy array
            indices = np.arange(feat_value.shape[0])
            np.random.shuffle(indices)
            shuffled_feat_dict[feat_name] = feat_value[indices]
        elif isinstance(feat_value, dict):
            # Shuffle dictionary entries by shuffling keys
            values = list(feat_value.values())
            random.shuffle(values)
            shuffled_feat_dict[feat_name] = dict(zip(list(feat_value.keys()), values))
        else:
            raise TypeError(f"Unsupported value type for key '{feat_name}': {type(feat_value)}")

    return shuffled_feat_dict