from os.path import join
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