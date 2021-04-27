import os
os.environ['R_HOME'] = '/home/tasnina/miniconda3/envs/decagon/lib/R'
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def compute_roc_prc(output_df):
    prroc = importr('PRROC')
    prCurve = prroc.pr_curve(scores_class0 = FloatVector(list(output_df['predicted'].values)),
              weights_class0 = FloatVector(list(output_df['true'].values)))

    fpr, tpr, thresholds = roc_curve(y_true=output_df['true'],
                                     y_score=output_df['predicted'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=output_df['true'],
                                                      probas_pred=output_df['predicted'], pos_label=1)
    return prec, recall, fpr, tpr, prCurve[2][0], auc(fpr, tpr)
