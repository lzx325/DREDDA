import os
from os.path import join
from itertools import chain
import argparse
import yaml

import pandas as pd
import numpy as np

import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.manifold
import sklearn.linear_model
import sklearn.svm

import torch
import torch.utils.data

import dredda.model as model

import dredda.data as data
import dredda.test as test
import dredda.model as model
import dredda.train as train
from dredda.download import download_if_not_exist


def get_remote_files_manifest(remote_prefix, local_prefix, fp_list):
    remote_prefix = remote_prefix.rstrip("/")
    local_prefix = local_prefix.rstrip("/")
    fp_list = [fp.rstrip("/") for fp in fp_list]
    remote_files_manifest = []
    for fp in fp_list:
        remote_fp = join(remote_prefix, fp)
        local_fp = join(local_prefix, fp)
        remote_files_manifest.append((local_fp, remote_fp))
    return remote_files_manifest


remote_files_checksum = dict(
    [
        (
            "dataset/CRISPRi_LINCS_processed/CRISPRi_ae_expectation_logtrans_fs_1000-368.hdf",
            "def9aeb56cf8d9ccea448722f8084dcc",
        ),
        (
            "dataset/CRISPRi_LINCS_processed/level5_ae_expectation_logtrans_fs_1000-368.hdf",
            "9391ddb6ea2bdc4b020f7bc6c2bbb341",
        ),
        (
            "dataset/CRISPRi_LINCS_processed/level5_logtrans_DE_genes.hdf",
            "effba084b678f441d602b89975cd2d6b",
        ),
        (
            "dataset/LINCS/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt",
            "103b871d39e4872001a7c5a24bcd07e7",
        ),
        (
            "checkpoint/CRISPRi_LINCS-model-epoch_best-20210320.pt",
            "40533ad3c8f46a9d179097b558855134",
        ),
        (
            "dataset/CRISPRi_LINCS_processed/cells_batches_cluster.txt",
            "929c1d08601cbd278c6dca3765b37c05",
        ),
    ]
)

local_prefix = "download"
remote_prefix = "https://dredda.s3.amazonaws.com/"

remote_files_manifest = dict(
    zip(
        remote_files_checksum.keys(),
        get_remote_files_manifest(
            remote_prefix="https://dredda.s3.amazonaws.com/",
            local_prefix="download",
            fp_list=list(remote_files_checksum.keys()),
        ),
    )
)


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def main_train(args):
    if os.path.isdir(args.out_dir):
        raise ValueError("args.out_dir already exists")
    else:
        os.makedirs(args.out_dir)

    # save args
    with open(join(args.out_dir, "train--args.yaml"), "x") as f:
        yaml.dump(args.__dict__, f)

    # set seed
    seed_all(args.seed)

    # load data
    data_dict = data.get_CRISPRi_LINCS_dataset(
        feature_selection="MI_ae_expectation_1000"
    )
    X_source, Y_source, X_target, Y_target = (
        data_dict["X_source"],
        data_dict["Y_source"],
        data_dict["X_target"],
        data_dict["Y_target"],
    )
    source_dataset_name, target_dataset_name = (
        data_dict["source_dataset_name"],
        data_dict["target_dataset_name"],
    )
    X_source_train, X_source_test, Y_source_train, Y_source_test = (
        sklearn.model_selection.train_test_split(
            X_source, Y_source, test_size=0.2, random_state=123
        )
    )

    # train
    net = model.FCModelDualBranchAE(n_in_features=X_source.shape[1])
    net = net.cuda()
    trainer = train.DualBranchDATrainer(
        net,
        source_dataset_name,
        target_dataset_name,
        net.ae_encoder_t.parameters(),
        chain(
            net.ae_encoder_s.parameters(),
            net.ae_decoder.parameters(),
            net.feature.parameters(),
            net.class_classifier.parameters(),
            net.domain_classifier.parameters(),
        ),
        n_epochs=args.n_epochs,
        domain_adv_coeff=args.domain_adv_coeff,
        ddc_coeff=args.ddc_coeff,
        dual_training_epoch=args.dual_training_epoch,
        save_root=args.out_dir,
    )

    trainer.fit(
        X_source_train,
        Y_source_train,
        X_target,
        Y_target,
        X_source_val=X_source_test,
        Y_source_val=Y_source_test,
        save=True,
    )


def main_test(args):
    test_out_dir = join(args.out_dir, "test")
    os.makedirs(test_out_dir, exist_ok=True)

    # save args
    args_fp = join(test_out_dir, "test--args.yaml")
    with open(args_fp, "x") as f:
        yaml.dump(args.__dict__, f)

    level5_fs_df = pd.read_hdf(args.lincs_level5_fs_table_fp)
    level5_fs_df_T = level5_fs_df.T

    model_fp = args.ckpt_fp
    model_new = model.FCModelDualBranchAE(n_in_features=level5_fs_df_T.shape[1])
    model_new.load_state_dict(torch.load(model_fp))

    trainer_new = train.DualBranchDATrainer(
        model_new,
        args.source_dataset_name,
        args.target_dataset_name,
        model_new.ae_encoder_t.parameters(),
        chain(
            model_new.ae_encoder_s.parameters(),
            model_new.ae_decoder.parameters(),
            model_new.feature.parameters(),
            model_new.class_classifier.parameters(),
            model_new.domain_classifier.parameters(),
        ),
    )

    level5_pred_raw = trainer_new.predict(level5_fs_df_T.values, "source", True)[0]

    def pred_post_process(pred, feature_df):
        from scipy.special import softmax

        pred_df = pd.DataFrame(
            pred,
            index=feature_df.index,
            columns=["class_%d" % (i) for i in range(pred.shape[1])],
        )
        pred_df["prediction_expected_classes"] = pred_df.loc[
            :, ["class_2", "class_3"]
        ].sum(axis=1)
        pred_df = pred_df.apply(softmax, axis=1)
        return pred_df

    level5_pred_df = pred_post_process(level5_pred_raw, level5_fs_df_T)

    root_dir = "download/dataset"
    siginfo_fp = join(root_dir, "LINCS/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt")
    exp_results_fp = join(root_dir, "exp_results_single/exp_results_single.csv")
    deccode_plurip_fp = join(root_dir, "others/deccode_plurip.csv")

    lre = test.PredictionEvaluator(
        full_prediction_df=level5_pred_df,
        features_df=level5_fs_df_T,
        siginfo_fp=siginfo_fp,
        ref_score_fp=args.reference_score_fp,
        exp_results_fp=exp_results_fp,
        deccode_plurip_fp=deccode_plurip_fp,
    )

    lre.save_prediction(out_dir=test_out_dir)

    lre.save_evaluation_score(out_dir=test_out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--source_dataset_name", type=str, default="CRISPRi")
    train_parser.add_argument("--target_dataset_name", type=str, default="LINCS")
    train_parser.add_argument("--out_dir", type=str, default="train_dir/default_config")
    train_parser.add_argument("--n_epochs", type=int, default=200)
    train_parser.add_argument("--dual_training_epoch", type=int, default=150)
    train_parser.add_argument("--domain_adv_coeff", type=float, default=0.1)
    train_parser.add_argument("--ddc_coeff", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=41)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument(
        "--ckpt_fp",
        type=str,
        default=join(
            local_prefix, "checkpoint/CRISPRi_LINCS-model-epoch_best-20210320.pt"
        ),
    )
    test_parser.add_argument(
        "--lincs_level5_fs_table_fp",
        type=str,
        default=join(
            local_prefix,
            "dataset/CRISPRi_LINCS_processed/level5_ae_expectation_logtrans_fs_1000-368.hdf",
        ),
    )
    test_parser.add_argument("--source_dataset_name", type=str, default="CRISPRi")
    test_parser.add_argument("--target_dataset_name", type=str, default="LINCS")
    test_parser.add_argument("--out_dir", type=str, default="train_dir/default_config")
    test_parser.add_argument("--reference_score_fp", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    for k, (local_fp, remote_fp) in remote_files_manifest.items():
        checksum = remote_files_checksum[k]
        download_if_not_exist(local_fp, remote_fp, md5sum=checksum)
    args = parse_args()
    if args.command == "train":
        main_train(args)
    elif args.command == "test":
        main_test(args)
    else:
        raise ValueError("invalid command")
