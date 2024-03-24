
import os
import numpy as np
import pandas as pd

def get_CRISPRi_LINCS_dataset(feature_selection):
    root_dir="download/dataset/CRISPRi_LINCS_processed"
    if feature_selection=="MI":
        level5_fs_table_fp=os.path.join(root_dir,"level5_logtrans_fs.hdf")
        CRISPRi_fs_table_fp=os.path.join(root_dir,"CRISPRi_logtrans_fs.hdf")

    elif feature_selection=="DE":
        level5_fs_table_fp=os.path.join(root_dir,"level5_logtrans_DE_genes.hdf")
        CRISPRi_fs_table_fp=os.path.join(root_dir,"CRISPRi_logtrans_DE_genes.hdf")
        
    elif feature_selection=="MI_ae_expectation_1000":
        level5_fs_table_fp=os.path.join(root_dir,"level5_ae_expectation_logtrans_fs_1000-368.hdf")
        CRISPRi_fs_table_fp=os.path.join(root_dir,"CRISPRi_ae_expectation_logtrans_fs_1000-368.hdf")

    elif feature_selection=="MI_ae_expectation_2000":
        level5_fs_table_fp=os.path.join(root_dir,"level5_ae_expectation_logtrans_fs_2000-760.hdf")
        CRISPRi_fs_table_fp=os.path.join(root_dir,"CRISPRi_ae_expectation_logtrans_fs_2000-760.hdf")
        
    elif feature_selection=="MI_ae_expectation_500":
        level5_fs_table_fp=os.path.join(root_dir,"level5_ae_expectation_logtrans_fs_500-180.hdf")
        CRISPRi_fs_table_fp=os.path.join(root_dir,"CRISPRi_ae_expectation_logtrans_fs_500-180.hdf")

    level5_fs_df=pd.read_hdf(level5_fs_table_fp)
    CRISPRi_fs_df=pd.read_hdf(CRISPRi_fs_table_fp)
    CRISPRi_fs_values_df=CRISPRi_fs_df.iloc[:,1:]
    cluster_labels_df=pd.read_csv(os.path.join(root_dir,"cells_batches_cluster.txt"),sep=",",index_col=0)

    CRISPRi_fs_values_sample_df=CRISPRi_fs_values_df.copy()
    level5_fs_sample_df=level5_fs_df.sample(20000,axis=1,random_state=123)
    
    icindex=[CRISPRi_fs_values_df.columns.get_loc(i) for i in CRISPRi_fs_values_sample_df.columns]
    cluster_labels_sample_df=cluster_labels_df.iloc[icindex,:]

    CRISPRi_fs_values_T=CRISPRi_fs_values_df.values.T
    level5_fs_sample_values_T=level5_fs_sample_df.values.T

    X_source=CRISPRi_fs_values_T
    Y_source=cluster_labels_sample_df["cluster"].values
    X_target=level5_fs_sample_values_T

    return {
        'X_source':X_source,
        'Y_source':Y_source,
        'X_target':X_target,
        'Y_target':None,
        'source_dataset_name':"CRISPRi",
        'target_dataset_name':"LINCS"
    }


