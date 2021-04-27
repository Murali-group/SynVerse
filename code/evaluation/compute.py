import os
os.environ['R_HOME'] = '/home/tasnina/miniconda3/envs/decagon/lib/R'
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pandas as pd

def compute_roc_prc(output_df):
    prroc = importr('PRROC')
    prCurve = prroc.pr_curve(scores_class0 = FloatVector(list(output_df['predicted'].values)),
              weights_class0 = FloatVector(list(output_df['true'].values)))

    fpr, tpr, thresholds = roc_curve(y_true=output_df['true'],
                                     y_score=output_df['predicted'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=output_df['true'],
                                                      probas_pred=output_df['predicted'], pos_label=1)
    return prec, recall, fpr, tpr, prCurve[2][0], auc(fpr, tpr)

def compute_prec_rec_at_each_rank(df, filename, force_run=True):
    ## input: df => contains  columns: 'drug_1', 'drug_2', 'cell_line','predicted', 'true' (both 1 and 0 labled pairs are present).

    if(not os.path.exists(filename) | force_run==True):
        print('compute prec recall')
        #sort df according to descending order of prediction score
        df = df[['drug_1', 'drug_2', 'cell_line', 'predicted', 'true']].sort_values(by=['predicted'], ascending=False)
        #reset index set index to be 0-n. so the the new index can be used as rank
        # df['rank'] = pd.Series(range(0, len(df)))
        df = df.reset_index()
        precision=[]
        recall=[]
        TP = 0
        FP = 0
        total_positive = len(df[df['true'] == 1])
        for index, row in df.iterrows():
            TP = TP + row['true']
            FP = (index+1) - TP
            precision.append(TP/float(TP+FP))
            recall.append(TP/float(total_positive))

        df['precision'] = pd.Series(precision)
        df['recall'] = pd.Series(recall)
        df = df[['drug_1', 'drug_2', 'cell_line', 'predicted', 'true', 'precision','recall']]

        #change filename
        # filename = 'dummy.csv'
        df.to_csv(filename, sep='\t')
    else:
        df = pd.read_csv(filename, sep='\t')
        precision = list(df['precision'])
        recall = list(df['recall'])
    return precision, recall


def compute_early_prec(df, recall_val, e_prec_saving_file, force_run):
    precision, recall = compute_prec_rec_at_each_rank(df,e_prec_saving_file,  force_run)
    i = 0
    while True:
        if recall[i] >= recall_val:
            return precision[i]
        i+=1








