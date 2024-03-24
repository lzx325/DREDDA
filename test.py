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

def softmax_normalization(level5_pred):
    level5_pred=level5_pred.apply(softmax,axis=1)
    return level5_pred

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

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item

    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).

    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean reciprocal rank
    """
    rs = [np.asarray(r).nonzero()[0] for r in rs]

    return np.mean([ 
        np.mean([1. / (r[i] + 1) for i in range(len(r))]) 
        if r.size else 0. 
        for r in rs
    ])

def precision_at_k(r, k):
    """Score is precision @ k

    Relevance is binary (nonzero is relevant).

    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k


    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Precision @ k

    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)

    Relevance is binary (nonzero is relevant).

    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    # the total number of ground truth positive will be the number of positive elements in r
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision

    Relevance is binary (nonzero is relevant).

    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Normalized discounted cumulative gain
    """
    # full r is sorted
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

class PredictionEvaluator(object):
    def __init__(
        self,
        full_prediction_df,
        features_df,
        siginfo_df,
        expected_classes = ["class_2","class_3"],
        ref_score_fp=None,
        exp_results_fp=None,
        deccode_plurip_fp=None
        
    ):
        # root_dir="download/dataset"
        # level5_fs_table_fp=join(root_dir,"CRISPRi_LINCS_processed/level5_logtrans_DE_genes.hdf")
        # siginfo_fp = join(root_dir,"LINCS/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt")
        # self.level5_fs_df_T=pd.read_hdf(level5_fs_table_fp).T
        # siginfo_df=pd.read_csv(siginfo_fp,sep='\t',index_col=0)
        
        self.df = dict()
        assert set(full_prediction_df.index)==set(features_df.index)
        full_prediction_df = softmax_normalization(full_prediction_df)
        self.expected_classes = expected_classes
        self.df["prediction"]=full_prediction_df
        self.df["features"]=features_df
        self.df["siginfo"]=siginfo_df
        
        
        if ref_score_fp:
            # evaluation results
            score1 = pd.read_csv(ref_score_fp, index_col=0).iloc[:, 0]
            self.df["ref_score"]=score1

        if exp_results_fp:
            # evaluation results
            exp_results = pd.read_csv(exp_results_fp, index_col=0)
            self.df["exp_results"]=exp_results
        
        if deccode_plurip_fp:
            # evaluation results
            deccode_plurip = pd.read_csv(deccode_plurip_fp, index_col=0)
            deccode_plurip_cell_agg = deccode_plurip.groupby(["pert_name","cell_id"]).mean()
            deccode_plurip_mean = deccode_plurip_cell_agg.reset_index().groupby(["pert_name"]).mean()
            self.df["deccode_plurip_cell_agg"]=deccode_plurip_cell_agg
            self.df["deccode_plurip_mean"]=deccode_plurip_mean

        self.compute_DREDDA_score()

    def compute_DREDDA_score(self):
        prediction_expected_classes=self.df["prediction"].loc[:,self.expected_classes].sum(axis=1)
        self.df["prediction"]["prediction_expected_classes"]=prediction_expected_classes

        prediction_df_copy=self.df["prediction"].copy()
        prediction_df_copy.insert(0,"pert_name",self.df["siginfo"]["pert_iname"])
        prediction_df_copy.insert(1,"pert_id",self.df["siginfo"]["pert_id"])
        prediction_df_copy.insert(2,"cell_id",self.df["siginfo"]["cell_id"])

        prediction_pert_cell_agg=prediction_df_copy.groupby(["pert_name","cell_id"]).mean()
        prediction_pert_mean=prediction_pert_cell_agg.reset_index().groupby(["pert_name"]).mean()

        self.df.update({
            'prediction_pert_cell_agg':prediction_pert_cell_agg,
            'prediction_pert_mean':prediction_pert_mean
        })
    
    def save_prediction(self,out_dir,n_drugs=30):
        os.makedirs(out_dir,exist_ok=True)

        # unformatted output
        if False:
            import pickle as pkl
            df_dict_fp = os.path.join(out_dir,"df_dict.pkl")
            with open(df_dict_fp,"wb") as f:
                pkl.dump(self.df,f)

        # formatted output
        n_drugs=30
        pert_id_by_iname=self.df["siginfo"].query("pert_type=='trt_cp'").groupby(["pert_iname"])["pert_id"].first()
        concat_df = pd.merge(self.df["prediction_pert_mean"],pert_id_by_iname,left_on="pert_name",right_index=True,how="left").reset_index()
        concat_df["pert_id_and_name"]=concat_df.apply(lambda x: "%s(%s)"%(x["pert_id"],x["pert_name"]), axis=1)

        concat_df=concat_df.sort_values("prediction_expected_classes",ascending=False).set_index("pert_name")
        concat_df.index.name=None
        

        # save full list
        fields = ["prediction_expected_classes","pert_id","pert_id_and_name"]
        concat_df[fields].to_csv(os.path.join(out_dir,"full_list.csv"),sep=',')
        # save top30
        with open(os.path.join(out_dir,"top30.txt"),"x") as f:
            print('\n'.join(concat_df["pert_id_and_name"].iloc[:n_drugs].values),file=f)
        # save top50
        with open(os.path.join(out_dir,"top50.txt"),"x") as f:
            print('\n'.join(concat_df["pert_id_and_name"].iloc[:50].values),file=f)
        # save bottom30
        with open(os.path.join(out_dir,"bottom30.txt"),"x") as f:
            print('\n'.join(concat_df["pert_id_and_name"].iloc[-n_drugs:].values),file=f)
        # save middle30
        with temp_seed(10):
            with open(os.path.join(out_dir,"middle30_rand1.txt"),"x") as f:
                print('\n'.join(concat_df["pert_id_and_name"].iloc[n_drugs:-n_drugs].sample(n_drugs).values),file=f)

            with open(os.path.join(out_dir,"middle30_rand2.txt"),"x") as f:
                print('\n'.join(concat_df["pert_id_and_name"].iloc[n_drugs:-n_drugs].sample(n_drugs).values),file=f)
    
    def evaluate_with_reference_prediction(self):
        # evaluation functions
        def spearman_correlation(scores1: pd.Series, scores2: pd.Series):
            assert set(scores1.index)==set(scores2.index)
            scores1 = scores1.loc[scores2.index]
            corr,p= scipy.stats.spearmanr(scores1, scores2)
            return corr.item()

        def overlap_count(scores1: pd.Series, scores2: pd.Series, topn=50):
            assert set(scores1.index)==set(scores2.index)
            scores1 = scores1.sort_values(ascending=False)
            scores2 = scores2.sort_values(ascending=False)
            topn = min(len(scores1), len(scores2), topn)
            overlap = set(scores1.iloc[:topn].index) & set(scores2.iloc[:topn].index)
            return len(overlap)
        

        if "ref_score" in self.df:
            score1 = self.df["ref_score"]
            score2 = self.df["prediction"]["prediction_pert_mean"]["prediction_expected_classes"]
            ## spearman correlation
            corr = spearman_correlation(score1, score2)
            ## overlap count
            ocount_30 = overlap_count(score1, score2, topn=30)
            ocount_50 = overlap_count(score1, score2, topn=50)
            ## save result to yaml
            return ({
                "spearman_correlation":corr,
                "overlap_count_top30":ocount_30,
                "overlap_count_top50":ocount_50,
            })
        else:
            return dict()
    
    def IR_metrics(self,prediction_values,relevant_items):
        prediction_values = prediction_values.sort_values(ascending=False)
        ranked_relevance=pd.Series(np.zeros_like(prediction_values.values),index=prediction_values.index)
        ranked_relevance.loc[np.intersect1d(relevant_items,ranked_relevance.index)]=1
        mrr = mean_reciprocal_rank([ranked_relevance.values]).item()

        ndcg_50 = ndcg_at_k(ranked_relevance.values, 50).item()
        ndcg_100 = ndcg_at_k(ranked_relevance.values, 100).item()
        ndcg_150 = ndcg_at_k(ranked_relevance.values, 150).item()
        ndcg_200 = ndcg_at_k(ranked_relevance.values, 200).item()

        return {
            "mean_reciprocal_rank":mrr,
            "nDCG@50":ndcg_50,
            "nDCG@100":ndcg_100,
            "nDCG@150":ndcg_150,
            "nDCG@200":ndcg_200
        }

    def evaluate_with_other_datasets(self,dataset_name):
        if dataset_name == "exp_results" and "exp_results" in self.df:
            prediction_values=self.df["prediction_pert_mean"]["prediction_expected_classes"].sort_values(ascending=False)
            exp_results=self.df["exp_results"]
            exp_results["mean_of_count_total_area"]=(exp_results["count"]+exp_results["total_area"])/2
            exp_results = exp_results.sort_values("mean_of_count_total_area",ascending=True)
            ranked_relevance=pd.Series(np.zeros_like(prediction_values.values),index=prediction_values.index)
            metrics = self.IR_metrics(prediction_values,exp_results.index[:10])
            return metrics
        elif dataset_name == "deccode" and "deccode_plurip_mean" in self.df:
            prediction_values=self.df["prediction_pert_mean"]["prediction_expected_classes"].sort_values(ascending=False)
            deccode_plurip_mean=self.df["deccode_plurip_mean"]
            deccode_plurip_mean = deccode_plurip_mean.sort_values("DECCODE_score",ascending=True)
            ranked_relevance=pd.Series(np.zeros_like(prediction_values.values),index=prediction_values.index)
            metrics = self.IR_metrics(prediction_values,deccode_plurip_mean.index[:10])
            return metrics
        else:
            return dict()
        
    def evaluate_diversity(self):
        prediction_cat=self.df["prediction_pert_mean"][["class_0","class_1","class_2","class_3"]].values.argmax(axis=1)
        _, counts = np.unique(prediction_cat, return_counts=True)
        freq = counts / counts.sum()
        # entropy of the categories
        ent = scipy.stats.entropy(freq).item()
        max_ent = scipy.stats.entropy(np.ones_like(freq)/len(freq)).item()
        return {
            "entropy_ratio":ent/max_ent
        }
    
    def save_evaluation_score(self,out_dir):
        import yaml
        eval_result = dict()

        # eval using diversity
        eval_result.update(self.evaluate_diversity())

        # eval using ref_score
        eval_result.update(self.evaluate_with_reference_prediction())

        # eval using the IR metrics
        eval_result.update(( { k+"_exp_results":v for k,v in self.evaluate_with_other_datasets("exp_results").items() } ))
        eval_result.update(( { k+"_deccode":v for k,v in self.evaluate_with_other_datasets("deccode").items() } ))

        print(eval_result)

        if len(eval_result)>0:
            with open(os.path.join(out_dir,"eval_result.yaml"),"x") as f:
                yaml.dump(eval_result,f)