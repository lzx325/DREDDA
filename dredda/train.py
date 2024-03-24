import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import collections
import pickle as pkl
from pprint import pprint
from sklearn.metrics import confusion_matrix, roc_auc_score


class DualBranchDATrainer(object):
    """This class handles the training of a dual-branch domain adaptation model."""

    def __init__(
        self,
        model,
        source_dataset_name,
        target_dataset_name,
        dual_branch_parameter,
        other_parameter,
        save_root=None,
        batch_size=128,
        n_epochs=200,
        dual_training_epoch=200,
        alpha=1,
        domain_adv_coeff=1e-1,
        ddc_coeff=1e-2,
        ddc_features="c_fc2",
        target_lr=1e-5,
        other_lr=1e-4,
        clip_value=1,
    ):
        """
        :param model: The model to be trained.
        :param source_dataset_name: The name of the source dataset.
        :param target_dataset_name: The name of the target dataset.
        :param dual_branch_parameter: The parameters of the dual branch of the model, which will be handled differently during training.
        :param other_parameter: The parameters of the other parts of the model.
        :param save_root: The root directory to save the model and scores.
        :param batch_size: The batch size.
        :param n_epochs: The number of epochs.
        :param dual_training_epoch: The epoch when the dual branch of the target encoder will be trained.
        :param alpha: The coefficient for the gradient reversal layer.
        :param domain_adv_coeff: The coefficient for the domain adversarial loss.
        :param ddc_coeff: The coefficient for the DDC loss.
        :param ddc_features: The name of the layer to return the features for the DDC loss.
        :param target_lr: The learning rate for the dual branch of the target encoder.
        :param other_lr: The learning rate for the other parts of the model.
        :param clip_value: The value to clip the gradient norm.
        """
        self.model = model
        self.other_lr = other_lr
        self.target_lr = target_lr
        self.dual_branch_optimizer = optim.Adam(
            [{"params": dual_branch_parameter}], lr=self.target_lr
        )
        self.other_optimizer = optim.Adam(
            [
                {
                    "params": other_parameter,
                }
            ],
            lr=self.other_lr,
        )
        self.device = next(self.model.parameters()).device

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dual_training_epoch = dual_training_epoch
        self.alpha = alpha
        self.domain_adv_coeff = domain_adv_coeff
        self.ddc_coeff = ddc_coeff
        self.ddc_features = ddc_features
        self.source_dataset_name = source_dataset_name
        self.target_dataset_name = target_dataset_name
        self.model_root = save_root
        self.clip_value = clip_value

    def fit(
        self,
        X_source,
        Y_source,
        X_target,
        Y_target,
        X_source_val=None,
        Y_source_val=None,
        n_epochs=None,
        save=True,
    ):
        """
        Fit the model.
        :param X_source: The source data.
        :param Y_source: The source labels.
        :param X_target: The target data.
        :param Y_target: The target labels, which could be None.
        :param X_source_val: The source validation data.
        :param Y_source_val: The source validation labels.
        :param n_epochs: The number of epochs.
        :param save: Whether to save the model and scores.
        """
        from dredda.helpers import iprint, iprint_dict

        if n_epochs is None:
            n_epochs = self.n_epochs
        if np.min(Y_source) == 1:
            Y_source = Y_source - 1
        if np.min(Y_target) == 1:
            Y_target = Y_target - 1
        if Y_source_val is not None and np.min(Y_source_val) == 1:
            Y_source_val = Y_source_val - 1
        assert (
            np.min(Y_source) == 0
            and (Y_target is None or np.min(Y_target) == 0)
            and (Y_source_val is None or np.min(Y_source_val) == 0)
        )

        X_source_tensor = torch.FloatTensor(X_source)
        Y_source_tensor = torch.LongTensor(Y_source)
        X_target_tensor = torch.FloatTensor(X_target)

        if Y_target is not None:
            Y_target_tensor = torch.LongTensor(Y_target)

        ds_source = torch.utils.data.TensorDataset(X_source_tensor, Y_source_tensor)
        if Y_target is not None:
            ds_target = torch.utils.data.TensorDataset(X_target_tensor, Y_target_tensor)
        else:
            ds_target = torch.utils.data.TensorDataset(X_target_tensor)

        dataloader_source = torch.utils.data.DataLoader(
            ds_source, batch_size=self.batch_size, shuffle=True
        )
        dataloader_target = torch.utils.data.DataLoader(
            ds_target, batch_size=self.batch_size, shuffle=True
        )
        iprint("=============== Training ===============")
        iprint(
            "Length of dataloaders",
            "source:",
            len(dataloader_source),
            "target:",
            len(dataloader_target),
            ident=0,
        )
        iprint(
            "Parameters:",
            "alpha = %.4f, domain_adv_coeff = %.4f, ddc_coeff = %.4f, ddc_features = %s"
            % (self.alpha, self.domain_adv_coeff, self.ddc_coeff, self.ddc_features),
            ident=0,
        )

        best_accu_s = 0.0
        best_accu_t = 0.0
        scores_dict = collections.defaultdict(list)

        source_domain_labels = np.zeros((len(X_source),), dtype=np.int64)
        score = self.score(X_source, Y_source, "source", False)
        accu_s = score["class_accuracy"]
        iprint("Before training:")
        iprint(
            "Accuracy on %s dataset: %f" % (self.source_dataset_name, accu_s), ident=1
        )
        iprint("Confusion matrix:", ident=1)
        iprint(score["confusion_matrix"])

        for epoch in range(n_epochs):
            self.model.train()

            # We first don't train the dual branch of the target encoder
            if epoch <= self.dual_training_epoch:
                self.model.set_dual_trainable(False)
            # After the dual training epoch, we train the dual branch of the target encoder
            if epoch == self.dual_training_epoch + 1:
                self.model.copy_params_primary_to_dual()
                self.model.set_dual_trainable(True)
            len_dataloader = min(len(dataloader_source), len(dataloader_target))
            data_source_iter = iter(dataloader_source)
            data_target_iter = iter(dataloader_target)
            cumulative_metrics = collections.defaultdict(float)
            for i in range(len_dataloader):
                data_source = data_source_iter.next()
                s_data, s_label = data_source
                s_domain_label = torch.zeros(len(s_data)).long()

                data_target = data_target_iter.next()
                if Y_target is not None:
                    t_data, _ = data_target
                else:
                    (t_data,) = data_target
                t_domain_label = torch.ones(len(t_data)).long()

                self.other_optimizer.zero_grad()
                if epoch > self.dual_training_epoch:
                    self.dual_branch_optimizer.zero_grad()

                s_data = s_data.to(self.device)
                s_label = s_label.to(self.device)
                s_domain_label = s_domain_label.to(self.device)

                t_data = t_data.to(self.device)
                t_domain_label = t_domain_label.to(self.device)

                s_class_output, s_domain_output, s_ddc_features = self.model(
                    s_data,
                    "source",
                    use_dual_branch=epoch > self.dual_training_epoch,
                    alpha=self.alpha,
                    return_ddc_features=self.ddc_features,
                )
                _, t_domain_output, t_ddc_features = self.model(
                    t_data,
                    "target",
                    use_dual_branch=epoch > self.dual_training_epoch,
                    alpha=self.alpha,
                    return_ddc_features=self.ddc_features,
                )

                # task loss
                err_s_label = F.cross_entropy(s_class_output, s_label)

                # adv domain loss
                err_s_domain = F.cross_entropy(s_domain_output, s_domain_label)
                err_t_domain = F.cross_entropy(t_domain_output, t_domain_label)

                # apply the DDC loss
                def loss_ddc(f_of_X, f_of_Y):
                    bs1 = f_of_X.shape[0]
                    bs2 = f_of_Y.shape[0]
                    bs = min(bs1, bs2)
                    delta = f_of_X[:bs, :] - f_of_Y[:bs, :]
                    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
                    return loss

                err_ddc = loss_ddc(s_ddc_features, t_ddc_features)

                err = (
                    self.domain_adv_coeff * (err_t_domain + err_s_domain)
                    + self.ddc_coeff * (err_ddc)
                    + err_s_label
                )
                err.backward()
                clip_grad_norm_(self.model.parameters(), self.clip_value)
                if epoch > self.dual_training_epoch:
                    self.other_optimizer.step()
                    self.dual_branch_optimizer.step()
                else:
                    self.other_optimizer.step()

                cumulative_metrics["domain_loss_t_domain"] += self.domain_adv_coeff * (
                    err_t_domain.cpu().item() / len_dataloader
                )
                cumulative_metrics["domain_loss_s_domain"] += self.domain_adv_coeff * (
                    err_s_domain.cpu().item() / len_dataloader
                )
                cumulative_metrics["class_loss_s_domain"] += (
                    err_s_label.cpu().item() / len_dataloader
                )
                cumulative_metrics["ddc"] += (
                    self.ddc_coeff * err_ddc.cpu().item() / len_dataloader
                )
                cumulative_metrics["loss"] += err.cpu().item() / len_dataloader

            iprint()
            iprint("Epoch %d:" % (epoch + 1))
            iprint("Cumulative metrics:", ident=1)
            iprint_dict(cumulative_metrics, ident=2)
            iprint("On source train set:", ident=1)
            source_domain_labels = np.zeros((len(X_source),), dtype=np.int64)
            score_source_train = self.score(
                X_source, Y_source, "source", epoch > self.dual_training_epoch
            )
            iprint_dict(
                score_source_train,
                ident=2,
                keys=["class_accuracy", "class_loss", "domain_loss", "domain_accuracy"],
            )
            iprint("confusion matrix:", ident=2)
            iprint(score_source_train["confusion_matrix"], ident=0)

            for k, v in score_source_train.items():
                scores_dict[k + "_source_train"].append(v)
            iprint()

            iprint("On source val set:", ident=1)
            if X_source_val is not None:
                source_domain_labels = np.zeros((len(X_source_val),), dtype=np.int64)
                score_source_val = self.score(
                    X_source_val,
                    Y_source_val,
                    "source",
                    epoch > self.dual_training_epoch,
                )
                iprint_dict(
                    score_source_val,
                    ident=2,
                    keys=[
                        "class_accuracy",
                        "class_loss",
                        "domain_loss",
                        "domain_accuracy",
                    ],
                )
                iprint("confusion matrix:", ident=2)
                iprint(score_source_val["confusion_matrix"], ident=0)

                for k, v in score_source_val.items():
                    scores_dict[k + "_source_val"].append(v)
            iprint()
            iprint("On target set:", ident=1)
            target_domain_labels = np.ones((len(X_target),), dtype=np.int64)
            score_target = self.score(
                X_target, Y_target, "target", epoch > self.dual_training_epoch
            )
            iprint_dict(score_target, ident=2)

            for k, v in score_target.items():
                scores_dict[k + "_target"].append(v)
            if save and self.model_root is not None:
                score_fp = "{}/{}_{}-score.pkl".format(
                    self.model_root, self.source_dataset_name, self.target_dataset_name
                )

                with open(score_fp, "wb") as f:
                    pkl.dump(scores_dict, f)

                current_model_fp = "{}/{}_{}-model-epoch_current.pt".format(
                    self.model_root, self.source_dataset_name, self.target_dataset_name
                )
                torch.save(self.model.state_dict(), current_model_fp)

                accu_s = score_source_train["class_accuracy"]
                if Y_target is not None:
                    accu_t = score_target["class_accuracy"]
                if (Y_target is not None and accu_t > best_accu_t) or (
                    Y_target is None and accu_s > best_accu_s
                ):
                    best_accu_s = accu_s
                    if Y_target is not None:
                        best_accu_t = accu_t
                    best_model_fp = "{}/{}_{}-model-epoch_best.pt".format(
                        self.model_root,
                        self.source_dataset_name,
                        self.target_dataset_name,
                    )
                    torch.save(self.model.state_dict(), best_model_fp)

    def predict(self, X, domain, use_dual_branch, batch_size=None, return_score=True):
        if batch_size is None:
            batch_size = self.batch_size
        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        class_output_list = list()
        domain_output_list = list()
        device = next(iter(self.model.parameters())).device
        with torch.no_grad():
            for _, (X_batch,) in enumerate(loader):
                X_batch = X_batch.to(device)
                if use_dual_branch:
                    class_output, domain_output = self.model(
                        X_batch,
                        domain,
                        use_dual_branch,
                        alpha=self.alpha,
                        return_ddc_features=None,
                    )

                else:
                    class_output, domain_output = self.model(
                        X_batch,
                        None,
                        use_dual_branch,
                        alpha=self.alpha,
                        return_ddc_features=None,
                    )
                if return_score:
                    class_output = class_output
                else:
                    class_output = class_output.argmax(1)
                class_output_list.append(class_output)
                domain_output_list.append(domain_output)

        class_output_tensor = torch.cat(class_output_list, 0)
        domain_output_tensor = torch.cat(domain_output_list, 0)
        class_output_arr = class_output_tensor.cpu().numpy()
        domain_output_arr = domain_output_tensor.cpu().numpy()
        self.model.train()
        return class_output_arr, domain_output_arr

    def score(self, X, Y, domain, use_dual_branch):
        if domain == "source":
            domain_labels = np.zeros((len(X),), dtype=np.int64)
        elif domain == "target":
            domain_labels = np.ones((len(X),), dtype=np.int64)
        if Y is not None:
            if np.min(Y) == 1:
                Y = Y - 1
            elif np.min(Y) != 0:
                assert False

            class_output_arr, domain_output_arr = self.predict(
                X, domain, use_dual_branch, return_score=True
            )
            class_output_tensor = torch.FloatTensor(class_output_arr)
            Y_tensor = torch.LongTensor(Y)
            domain_labels_tensor = torch.LongTensor(domain_labels)
            domain_output_tensor = torch.FloatTensor(domain_output_arr)
            class_loss = F.cross_entropy(
                class_output_tensor, Y_tensor, reduction="mean"
            )
            class_output_idx_arr = class_output_arr.argmax(1)
            class_acc = np.mean(class_output_idx_arr == Y)
            domain_loss = F.cross_entropy(
                domain_output_tensor, domain_labels_tensor, reduction="mean"
            )
            domain_output_idx_arr = domain_output_tensor.argmax(1).numpy()
            domain_acc = np.mean(domain_output_idx_arr == domain_labels)

            if class_output_arr.shape[1] == 2:
                auc = roc_auc_score(Y, class_output_arr[:, 1])
                return {
                    "class_accuracy": class_acc,
                    "class_loss": class_loss,
                    "domain_loss": self.domain_adv_coeff * domain_loss,
                    "domain_accuracy": domain_acc,
                    "auc": auc,
                }
            else:
                cf = confusion_matrix(Y, class_output_idx_arr)
                return {
                    "class_accuracy": class_acc,
                    "class_loss": class_loss,
                    "domain_loss": self.domain_adv_coeff * domain_loss,
                    "domain_accuracy": domain_acc,
                    "confusion_matrix": cf,
                }

        else:
            _, domain_output_arr = self.predict(
                X, domain, use_dual_branch, return_score=True
            )
            domain_labels_tensor = torch.LongTensor(domain_labels)
            domain_output_tensor = torch.FloatTensor(domain_output_arr)
            domain_loss = F.cross_entropy(
                domain_output_tensor, domain_labels_tensor, reduction="mean"
            )
            domain_output_idx_arr = domain_output_tensor.argmax(1).numpy()
            domain_acc = np.mean(domain_output_idx_arr == domain_labels)
            return {
                "domain_loss": self.domain_adv_coeff * domain_loss,
                "domain_accuracy": domain_acc,
            }
