import os
import torch
import torch.utils.data
import data
from torchvision import datasets
import sklearn.model_selection
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import contextlib
from MulticoreTSNE import MulticoreTSNE as TSNE
import pandas as pd
import scipy.stats
from scipy.special import softmax
from scipy.stats import rankdata
from os.path import join

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def clone_state(net):
    params_dict=dict()
    for name,param in net.state_dict().items():
        params_dict[name]=param.detach().clone()
    return params_dict

def check_state_equivalence(param_dict1,param_dict2):
    for name, param in param_dict1.items():
        if torch.allclose(param,param_dict2[name]):
            print("%s: same"%(name))
        else:
            max_diff=torch.max(torch.abs(param-param_dict2[name]))
            print("%s: different, max difference: %.5f"%(name,max_diff))

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

def softmax_normalization(level5_pred,level5_fs_df_T,qnorm=False):
    level5_pred=softmax(level5_pred,axis=1)
    level5_pred=level5_pred[:,2]+level5_pred[:,3]
    if qnorm:
        level5_pred=rankdata(level5_pred, 'average')/len(level5_pred)
    level5_pred_df=pd.DataFrame((level5_pred)[:,None],index=level5_fs_df_T.index)

    return level5_pred_df

def l1_norm_normalization(level5_pred,level5_fs_df_T,qnorm=False):
    level5_pred=level5_pred/np.abs(level5_pred).sum(axis=1,keepdims=True)
    level5_pred=level5_pred[:,2]+level5_pred[:,3]
    if qnorm:
        level5_pred=rankdata(level5_pred, 'average')/len(level5_pred)
    level5_pred_df=pd.DataFrame(level5_pred[:,None],index=level5_fs_df_T.index)
    return level5_pred_df

def l2_norm_normalization(level5_pred,level5_fs_df_T,qnorm=False):
    level5_pred=level5_pred/np.linalg.norm(level5_pred,axis=1,keepdims=True)
    level5_pred=level5_pred[:,2]+level5_pred[:,3]
    if qnorm:
        level5_pred=rankdata(level5_pred, 'average')/len(level5_pred)
    level5_pred_df=pd.DataFrame((level5_pred)[:,None],index=level5_fs_df_T.index)
    return level5_pred_df

class LINCSRankingEvaluator(object):
    def __init__(self,prediction_df):
        root_dir="download/dataset"
        level5_fs_table_fp=join(root_dir,"CRISPRi_LINCS_processed/level5_logtrans_DE_genes.hdf")
        self.level5_fs_df_T=pd.read_hdf(level5_fs_table_fp).T
        assert set(prediction_df.index)==set(self.level5_fs_df_T.index)

        siginfo_fp = join(root_dir,"LINCS/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt")
        siginfo_df=pd.read_csv(siginfo_fp,sep='\t',index_col=0)
        self.siginfo_df=siginfo_df
        self.prediction_df=prediction_df
        prediction_df_copy=prediction_df.copy()
        prediction_df_copy.insert(0,"pert_name",siginfo_df["pert_iname"])
        prediction_df_copy.insert(1,"cell_id",siginfo_df["cell_id"])

        prediction_mean_df=prediction_df_copy.groupby(["pert_name","cell_id"]).mean().reset_index()
        prediction_grouped_df=prediction_mean_df.groupby(["pert_name"]).mean()

        self.df={
            'prediction':prediction_grouped_df,
        }