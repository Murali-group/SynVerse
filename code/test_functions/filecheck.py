import pandas
import pickle


cross_val_dir = 
pos_train_test_val_file = cross_val_dir + 'pos_train_test_val.pkl'
neg_train_test_val_file = cross_val_dir + 'neg_train_test_val.pkl'

with open(neg_train_test_val_file, 'rb') as handle:
    type_wise_neg_cross_folds[cross_val_type] = pickle.load(handle)