import torch.nn as nn
import torch.nn.functional as F
import torch
from dredda.functions import ReverseLayerF
from collections import OrderedDict


class FCModelDualBranchAE(nn.Module):
    """
    The main model for domain adaptation. This model is a dual-branch autoencoder with a feature extractor, a task classifier, and a domain adversarial classifier.
    """

    def __init__(self, n_in_features, n_out_classes=4):
        """
        :param n_in_features: The number of input features.
        :param n_out_classes: The number of output classes.
        """
        super().__init__()
        self.n_in_features = n_in_features
        self.n_out_classes = n_out_classes

        # definition of source domain encoder
        self.ae_encoder_s = nn.Sequential(
            OrderedDict(
                {
                    "ae_encoder_s_fc1": nn.Linear(n_in_features, 100),
                    "ae_encoder_s_relu1": nn.ReLU(True),
                    "ae_encoder_s_fc2": nn.Linear(100, 50),
                    "ae_encoder_s_relu2": nn.ReLU(True),
                }
            )
        )

        # definition of target domain encoder
        self.ae_encoder_t = nn.Sequential(
            OrderedDict(
                {
                    "ae_encoder_t_fc1": nn.Linear(n_in_features, 100),
                    "ae_encoder_t_relu1": nn.ReLU(True),
                    "ae_encoder_t_fc2": nn.Linear(100, 50),
                    "ae_encoder_t_relu2": nn.ReLU(True),
                }
            )
        )

        # definition of the shared decoder
        self.ae_decoder = nn.Sequential(
            OrderedDict(
                {
                    "ae_decoder_fc1": nn.Linear(50, 100),
                    "ae_decoder_relu1": nn.ReLU(True),
                    "ae_decoder_fc2": nn.Linear(100, n_in_features),
                }
            )
        )

        # definition of the feature extractor
        self.feature = nn.Sequential(
            OrderedDict(
                {
                    "f_fc1": nn.Linear(n_in_features, 100),
                    "f_relu1": nn.ReLU(True),
                    "f_fc2": nn.Linear(100, 80),
                    "f_relu2": nn.ReLU(True),
                    "f_fc3": nn.Linear(80, 60),
                    "f_relu3": nn.ReLU(True),
                }
            )
        )

        # definition of the task classifier
        self.class_classifier = nn.Sequential(
            OrderedDict(
                {
                    "c_fc1": nn.Linear(60, 40),
                    "c_relu1": nn.ReLU(True),
                    "c_fc2": nn.Linear(40, 20),
                    "c_relu2": nn.ReLU(True),
                    "c_fc3": nn.Linear(20, n_out_classes),
                }
            )
        )

        # definition of the adv domain classifier
        self.domain_classifier = nn.Sequential(
            OrderedDict(
                {
                    "d_fc1": nn.Linear(60, 40),
                    "d_relu1": nn.ReLU(True),
                    "d_fc2": nn.Linear(40, 2),
                }
            )
        )

    def copy_params_primary_to_dual(self):
        """Copy the parameters of the primary branch to the dual branch of the autoencoder."""
        self.zero_grad()
        for i in range(len(self.ae_encoder_s)):
            if isinstance(self.ae_encoder_s[i], nn.Linear):
                self.ae_encoder_t[i].weight.data[:] = self.ae_encoder_s[i].weight[:]
                self.ae_encoder_t[i].bias.data[:] = self.ae_encoder_s[i].bias[:]

    def set_dual_trainable(self, trainable):
        """Set the dual branch of the autoencoder to be trainable or not."""
        for p in self.ae_encoder_t.parameters():
            p.requires_grad_(trainable)

    def forward(
        self, input_data, domain, use_dual_branch, alpha, return_ddc_features=None
    ):
        """
        Forward pass of the model.
        :param input_data: The input data.
        :param domain: Either "source" or "target". The domain of the input data.
        :param use_dual_branch: Whether to use the dual branch of the autoencoder.
        :param alpha: The coefficient for the gradient reversal layer.
        :param return_ddc_features: The name of the layer to return the features for the domain adversarial classifier.
        """

        if return_ddc_features is not None:
            assert return_ddc_features in self.class_classifier._modules
        if use_dual_branch:
            # use the source domain encoder if the domain is source
            if domain == "source":
                feature = self.ae_encoder_s(input_data)
            # use the target domain encoder if the domain is target
            elif domain == "target":
                feature = self.ae_encoder_t(input_data)
            else:
                assert False
        else:
            # use the source domain encoder for all data if not using the dual branch
            feature = self.ae_encoder_s(input_data)
        # apply the shared decoder
        feature = self.ae_decoder(feature)

        # apply the feature extractor
        feature = self.feature(feature)

        # apply the gradient reversal layer
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = feature
        ddc_features = None

        # sequentially apply the task classifier and get the intermediates
        for k, v in self.class_classifier._modules.items():
            class_output = v(class_output)
            if k == return_ddc_features:
                ddc_features = class_output

        # apply the domain adversarial classifier
        domain_output = self.domain_classifier(reverse_feature)
        if return_ddc_features:
            return class_output, domain_output, ddc_features
        else:
            return class_output, domain_output

    def transform(self, X, domain, use_dual_branch, layer="fc", batch_size=128):
        """get the network representation from the model."""
        assert layer in ("feature", "fc")

        self.eval()
        device = next(self.parameters()).device
        X_tensor = torch.FloatTensor(X)

        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        X_transformed_tensor_list = list()

        def fc_transformation(fc, val):
            fc_layer_name = "c_fc2"
            for k, m in fc._modules.items():
                val = m(val)
                if k == fc_layer_name:
                    break
            return val

        with torch.no_grad():
            for i, (X_batch,) in enumerate(loader):
                X_batch = X_batch.to(device)
                if use_dual_branch and domain == "target":
                    X_batch_transformed_tensor = self.ae_encoder_t(X_batch)
                else:
                    X_batch_transformed_tensor = self.ae_encoder_s(X_batch)
                X_batch_transformed_tensor = self.ae_decoder(X_batch_transformed_tensor)
                X_batch_transformed_tensor = self.feature(X_batch_transformed_tensor)
                if layer == "fc":
                    X_batch_transformed_tensor = fc_transformation(
                        self.class_classifier, X_batch_transformed_tensor
                    )
                X_transformed_tensor_list.append(X_batch_transformed_tensor)
            X_transformed_tensor = torch.cat(X_transformed_tensor_list, dim=0)
        X_transformed_arr = X_transformed_tensor.cpu().numpy()
        self.train()
        return X_transformed_arr
