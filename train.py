
import random
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import collections
import pickle as pkl
from pprint import pprint
from sklearn.metrics import confusion_matrix,roc_auc_score


class DATrainer(object):
    def __init__(
        self,
        model,
        optimizer_fn,
        optimizer_params,
        source_dataset_name,
        target_dataset_name,
        ):
        self.model=model
        self.optimizer=optimizer_fn(self.model.parameters(),**optimizer_params)
        self.clip_value=1
        self.device=next(self.model.parameters()).device
        self.batch_size=128
        self.n_epochs=200
        self.alpha = 1
        self.domain_adv_coeff = 1e-1
        self.ddc_coeff= 1e-2
        self.ddc_features="c_fc2"
        self.source_dataset_name=source_dataset_name
        self.target_dataset_name=target_dataset_name
        self.model_root="./models"

    def fit(
        self,
        X_source,
        Y_source,
        X_target,
        Y_target,
        X_source_val=None,
        Y_source_val=None,
        n_epochs=None,
        save=True
    ):
        from sklearn.utils.class_weight import compute_class_weight
        
        if n_epochs is None:
            n_epochs=self.n_epochs
        if np.min(Y_source)==1:
            Y_source=Y_source-1
        if np.min(Y_target)==1:
            Y_target=Y_target-1
        if Y_source_val is not None and np.min(Y_source_val)==1:
            Y_source_val=Y_source_val-1
        assert np.min(Y_source)==0 and (Y_target is None or np.min(Y_target)==0) and (Y_source_val is None or np.min(Y_source_val)==0)

        X_source_tensor=torch.FloatTensor(X_source)
        Y_source_tensor=torch.LongTensor(Y_source)
        X_target_tensor=torch.FloatTensor(X_target)

        class_weight = compute_class_weight(
            "balanced",
            classes=np.unique(np.concatenate([Y_source,Y_target])),
            y=np.concatenate([Y_source,Y_target])
        )

        if Y_target is not None:
            Y_target_tensor=torch.LongTensor(Y_target)
        
        ds_source=torch.utils.data.TensorDataset(X_source_tensor,Y_source_tensor)
        if Y_target is not None:
            ds_target=torch.utils.data.TensorDataset(X_target_tensor,Y_target_tensor)
        else:
            ds_target=torch.utils.data.TensorDataset(X_target_tensor)

        dataloader_source=torch.utils.data.DataLoader(ds_source,batch_size=self.batch_size,shuffle=True)
        dataloader_target=torch.utils.data.DataLoader(ds_target,batch_size=self.batch_size,shuffle=True)
        print("Length of dataloaders:")
        print(len(dataloader_source), len(dataloader_target))
        print("Parameters:")
        print("alpha=%.4f,domain_adv_coeff=%.4f,ddc_coeff=%.4f,ddc_features=%s"%(self.alpha,self.domain_adv_coeff,self.ddc_coeff,self.ddc_features))

        best_accu_s=0.0
        best_accu_t=0.0
        scores_dict=collections.defaultdict(list)


        print("before everything")
        source_domain_labels=np.zeros((len(X_source),),dtype=np.int64)
        score = self.score(X_source,Y_source,"source")
        accu_s=score["class_accuracy"]
        print('Accuracy of the %s dataset: %f' % (self.source_dataset_name, accu_s))
        print("confusion matrix:")
        print(score["confusion_matrix"])


        for epoch in range(n_epochs):
            self.model.train()
            len_dataloader = min(len(dataloader_source), len(dataloader_target))
            data_source_iter = iter(dataloader_source)
            data_target_iter = iter(dataloader_target)
            cumulative_metrics=collections.defaultdict(float)
            for i in range(len_dataloader):
                data_source=data_source_iter.next()
                s_img,s_label=data_source
                s_domain_label = torch.zeros(len(s_img)).long()

                data_target = data_target_iter.next()
                if Y_target is not None:
                    t_img, _ = data_target
                else:
                    t_img, = data_target
                t_domain_label = torch.ones(len(t_img)).long()

                self.optimizer.zero_grad()

                s_img=s_img.to(self.device)
                s_label=s_label.to(self.device)
                s_domain_label=s_domain_label.to(self.device)
                
                t_img=t_img.to(self.device)
                t_domain_label=t_domain_label.to(self.device)
                
                img=torch.cat([s_img,t_img],0)

                class_output,domain_output,ddc_features=self.model(img,alpha=self.alpha,return_ddc_features=self.ddc_features)
                s_class_output=class_output[:len(s_img)]
                s_domain_output=domain_output[:len(s_img)]
                t_domain_output=domain_output[len(s_img):]
                s_ddc_features=ddc_features[:len(s_img)]
                t_ddc_features=ddc_features[len(s_img):]

                print(class_weight)
                err_s_label=F.cross_entropy(s_class_output,s_label,weight=class_weight)
                err_s_domain=F.cross_entropy(s_domain_output,s_domain_label)

                err_t_domain=F.cross_entropy(t_domain_output,t_domain_label)
                

                def loss_ddc(f_of_X, f_of_Y):
                    bs1=f_of_X.shape[0]
                    bs2=f_of_Y.shape[0]
                    bs=min(bs1,bs2)
                    delta = f_of_X[:bs,:] - f_of_Y[:bs,:]
                    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
                    return loss

                err_ddc=loss_ddc(s_ddc_features,t_ddc_features)

                err = self.domain_adv_coeff * (err_t_domain + err_s_domain) + self.ddc_coeff*(err_ddc) + err_s_label
                err.backward()
                clip_grad_norm_(self.model.parameters(),self.clip_value)
                self.optimizer.step()

                cumulative_metrics["domain_loss_t_domain"]+=self.domain_adv_coeff*(err_t_domain.cpu().item()/len_dataloader)
                cumulative_metrics["domain_loss_s_domain"]+=self.domain_adv_coeff*(err_s_domain.cpu().item()/len_dataloader)
                cumulative_metrics["class_loss_s_domain"]+=err_s_label.cpu().item()/len_dataloader
                cumulative_metrics["ddc"]+=self.ddc_coeff*err_ddc.cpu().item()/len_dataloader
                cumulative_metrics["loss"]+=err.cpu().item()/len_dataloader

            print()
            print("Epoch %d"%(epoch+1))

            pprint(cumulative_metrics)
            print("On source train set: ")
            source_domain_labels=np.zeros((len(X_source),),dtype=np.int64)
            score_source_train = self.score(X_source,Y_source,"source")
            for k,v in score_source_train.items():
                print(f"\t{k}:\n\t{v}")
                scores_dict[k+"_source_train"].append(v)
            print()
            print("On source val set")
            if X_source_val is not None:
                source_domain_labels=np.zeros((len(X_source_val),),dtype=np.int64)
                score_source_val = self.score(X_source_val,Y_source_val,"source")
                for k,v in score_source_val.items():
                    print(f"\t{k}:\n\t{v}")
                    scores_dict[k+"_source_val"].append(v)

            print()
            print("On target set")
            target_domain_labels=np.ones((len(X_target),),dtype=np.int64)
            score_target = self.score(X_target,Y_target,"target")
            for k,v in score_target.items():
                print(f"\t{k}:\n{v}")
                scores_dict[k+"_target"].append(v)
            if save:
                score_fp='{}/{}_{}-score.pkl'.format(self.model_root,self.source_dataset_name,self.target_dataset_name)

                with open(score_fp,'wb') as f:
                    pkl.dump(scores_dict,f)

                current_model_fp='{}/{}_{}-model-epoch_current.pth'.format(self.model_root,self.source_dataset_name,self.target_dataset_name)
                torch.save(self.model, current_model_fp)
                

                accu_s=score_source_train["class_accuracy"]
                if Y_target is not None:
                    accu_t=score_target["class_accuracy"]
                if (Y_target is not None and accu_t > best_accu_t) or (Y_target is None and accu_s>best_accu_s):
                    best_accu_s = accu_s
                    if Y_target is not None:
                        best_accu_t = accu_t
                    best_model_fp='{}/{}_{}-model-epoch_best.pth'.format(self.model_root,self.source_dataset_name,self.target_dataset_name)
                    torch.save(self.model, best_model_fp)
            
    def predict(self,X,batch_size=None,return_score=True):
        if batch_size is None:
            batch_size=self.batch_size
        self.model.eval()
        X_tensor=torch.FloatTensor(X)
        dataset=torch.utils.data.TensorDataset(X_tensor)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        class_output_list=list()
        domain_output_list=list()
        device=next(iter(self.model.parameters())).device
        with torch.no_grad():
            for _,(X_batch,) in enumerate(loader):
                X_batch=X_batch.to(device)
                class_output,domain_output=self.model(X_batch,alpha=self.alpha,return_ddc_features=None)
                if return_score:
                    class_output=class_output
                else:
                    class_output=class_output.argmax(1)
                class_output_list.append(class_output)
                domain_output_list.append(domain_output)

        class_output_tensor=torch.cat(class_output_list,0)
        domain_output_tensor=torch.cat(domain_output_list,0)
        class_output_arr=class_output_tensor.cpu().numpy()
        domain_output_arr=domain_output_tensor.cpu().numpy()
        self.model.train()
        return class_output_arr,domain_output_arr

    def score(self,X,Y,domain):
        if domain=="source":
            domain_labels=np.zeros((len(X),),dtype=np.int64)
        elif domain=="target":
            domain_labels=np.ones((len(X),),dtype=np.int64)
        if Y is not None:
            if np.min(Y)==1:
                Y=Y-1
            elif np.min(Y)!=0:
                assert False

            class_output_arr,domain_output_arr=self.predict(X,return_score=True)
            class_output_tensor=torch.FloatTensor(class_output_arr)
            Y_tensor=torch.LongTensor(Y)
            domain_labels_tensor=torch.LongTensor(domain_labels)
            domain_output_tensor=torch.FloatTensor(domain_output_arr)
            class_loss=F.cross_entropy(class_output_tensor,Y_tensor,reduction="mean")
            class_output_idx_arr=class_output_arr.argmax(1)
            class_acc=np.mean(class_output_idx_arr==Y)
            domain_loss=F.cross_entropy(domain_output_tensor,domain_labels_tensor,reduction="mean")
            domain_output_idx_arr=domain_output_tensor.argmax(1).numpy()
            domain_acc=np.mean(domain_output_idx_arr==domain_labels)
            
            if class_output_arr.shape[1]==2:
                auc=roc_auc_score(Y,class_output_arr[:,1])
                return {
                    'class_accuracy':class_acc,
                    'class_loss':class_loss,
                    'domain_loss':self.domain_adv_coeff*domain_loss,
                    'domain_accuracy':domain_acc,
                    'auc':auc,
                }
            else:
                cf=confusion_matrix(Y,class_output_idx_arr)
                return {
                    'class_accuracy':class_acc,
                    'class_loss':class_loss,
                    'domain_loss':self.domain_adv_coeff*domain_loss,
                    'domain_accuracy':domain_acc,
                    'confusion_matrix':cf,
                }
            
        else:
            _,domain_output_arr=self.predict(X,return_score=True)
            domain_labels_tensor=torch.LongTensor(domain_labels)
            domain_output_tensor=torch.FloatTensor(domain_output_arr)
            domain_loss=F.cross_entropy(domain_output_tensor,domain_labels_tensor,reduction="mean")
            domain_output_idx_arr=domain_output_tensor.argmax(1).numpy()
            domain_acc=np.mean(domain_output_idx_arr==domain_labels)
            return {
                    'domain_loss':self.domain_adv_coeff*domain_loss,
                    'domain_accuracy':domain_acc,
            }
    def feature_importance(self,X):
        X_tensor=torch.FloatTensor(X)
        dataloader=torch.utils.data.DataLoader(X_tensor,batch_size=self.batch_size)
        scores=dict()
        scores["feature_grad"]=np.zeros((self.model.n_out_classes,X.shape[1]))
        scores["feature_grad_abs"]=np.zeros((self.model.n_out_classes,X.shape[1]))

        for _,X_batch in enumerate(dataloader):
            X_batch=X_batch.to(self.device)
            X_batch.requires_grad_()
            Y_pred,_=self.model(X_batch,alpha=self.alpha,return_ddc_features=None)
            Y_pred_sum=Y_pred.sum(0)
            for j in range(self.model.n_out_classes):
                X_batch_grad=torch.autograd.grad(Y_pred_sum[j],X_batch,retain_graph=True)[0]
                X_batch_grad_np=X_batch_grad.cpu().numpy()
                scores["feature_grad"][j]+=X_batch_grad_np.sum(axis=0)*(1/len(X))
                scores["feature_grad_abs"][j]+=np.abs(X_batch_grad_np).sum(axis=0)*(1/len(X))

        return scores
    
class DualBranchDATrainer(object):
    def __init__(
        self,
        model,
        source_dataset_name,
        target_dataset_name,
        dual_branch_parameter,
        other_parameter
        ):
        self.model=model
        self.other_lr=1e-4
        self.target_lr=1e-5
        self.dual_branch_optimizer=optim.Adam(
            [{
                'params':dual_branch_parameter
            }],
            lr=self.target_lr
        )
        self.other_optimizer=optim.Adam(
            [
                {
                    'params':other_parameter,
                }
            ],
            lr=self.other_lr
        )
        self.clip_value=1
        self.device=next(self.model.parameters()).device
        self.batch_size=128
        self.n_epochs=200
        self.dual_training_epoch=200
        self.alpha = 1
        self.domain_adv_coeff = 1e-1
        self.ddc_coeff= 1e-2
        self.ddc_features="c_fc2"
        self.source_dataset_name=source_dataset_name
        self.target_dataset_name=target_dataset_name
        self.model_root="./models"

    def fit(
        self,
        X_source,
        Y_source,
        X_target,
        Y_target,
        X_source_val=None,
        Y_source_val=None,
        n_epochs=None,
        save=True
    ):
        if n_epochs is None:
            n_epochs=self.n_epochs
        if np.min(Y_source)==1:
            Y_source=Y_source-1
        if np.min(Y_target)==1:
            Y_target=Y_target-1
        if Y_source_val is not None and np.min(Y_source_val)==1:
            Y_source_val=Y_source_val-1
        assert np.min(Y_source)==0 and (Y_target is None or np.min(Y_target)==0) and (Y_source_val is None or np.min(Y_source_val)==0)

        X_source_tensor=torch.FloatTensor(X_source)
        Y_source_tensor=torch.LongTensor(Y_source)
        X_target_tensor=torch.FloatTensor(X_target)

        
        if Y_target is not None:
            Y_target_tensor=torch.LongTensor(Y_target)
        
        ds_source=torch.utils.data.TensorDataset(X_source_tensor,Y_source_tensor)
        if Y_target is not None:
            ds_target=torch.utils.data.TensorDataset(X_target_tensor,Y_target_tensor)
        else:
            ds_target=torch.utils.data.TensorDataset(X_target_tensor)

        dataloader_source=torch.utils.data.DataLoader(ds_source,batch_size=self.batch_size,shuffle=True)
        dataloader_target=torch.utils.data.DataLoader(ds_target,batch_size=self.batch_size,shuffle=True)
        print("Length of dataloaders:")
        print(len(dataloader_source), len(dataloader_target))
        print("Parameters:")
        print("alpha=%.4f,domain_adv_coeff=%.4f,ddc_coeff=%.4f,ddc_features=%s"%(self.alpha,self.domain_adv_coeff,self.ddc_coeff,self.ddc_features))

        best_accu_s=0.0
        best_accu_t=0.0
        scores_dict=collections.defaultdict(list)


        print("before everything")
        source_domain_labels=np.zeros((len(X_source),),dtype=np.int64)
        score = self.score(X_source,Y_source,"source",False)
        accu_s=score["class_accuracy"]
        print('Accuracy of the %s dataset: %f' % (self.source_dataset_name, accu_s))
        print("confusion matrix:")
        print(score["confusion_matrix"])


        for epoch in range(n_epochs):
            # TODO: epoch that we copy the parameters
            self.model.train()
            if epoch<=self.dual_training_epoch:
                self.model.set_dual_trainable(False)
            if epoch==self.dual_training_epoch+1:
                self.model.copy_params_primary_to_dual()
                self.model.set_dual_trainable(True)
            len_dataloader = min(len(dataloader_source), len(dataloader_target))
            data_source_iter = iter(dataloader_source)
            data_target_iter = iter(dataloader_target)
            cumulative_metrics=collections.defaultdict(float)
            for i in range(len_dataloader):
                data_source=data_source_iter.next()
                s_img,s_label=data_source
                s_domain_label = torch.zeros(len(s_img)).long()

                data_target = data_target_iter.next()
                if Y_target is not None:
                    t_img, _ = data_target
                else:
                    t_img, = data_target
                t_domain_label = torch.ones(len(t_img)).long()
                # t_img=s_img
                # t_domain_label=s_domain_label
                self.other_optimizer.zero_grad()
                if epoch>self.dual_training_epoch:
                    self.dual_branch_optimizer.zero_grad()
                
                s_img=s_img.to(self.device)
                s_label=s_label.to(self.device)
                s_domain_label=s_domain_label.to(self.device)
                
                t_img=t_img.to(self.device)
                t_domain_label=t_domain_label.to(self.device)
                
                '''
                img=torch.cat([s_img,t_img],0)

                class_output,domain_output,ddc_features=self.model(img,alpha=self.alpha,return_ddc_features=self.ddc_features)
                s_class_output=class_output[:len(s_img)]
                s_domain_output=domain_output[:len(s_img)]
                t_domain_output=domain_output[len(s_img):]
                s_ddc_features=ddc_features[:len(s_img)]
                t_ddc_features=ddc_features[len(s_img):]
                '''
                s_class_output,s_domain_output,s_ddc_features=self.model(s_img,"source",use_dual_branch=epoch>self.dual_training_epoch,alpha=self.alpha,return_ddc_features=self.ddc_features)
                _,t_domain_output,t_ddc_features=self.model(t_img,"target",use_dual_branch=epoch>self.dual_training_epoch,alpha=self.alpha,return_ddc_features=self.ddc_features)


                err_s_label=F.cross_entropy(s_class_output,s_label)
                err_s_domain=F.cross_entropy(s_domain_output,s_domain_label)
                err_t_domain=F.cross_entropy(t_domain_output,t_domain_label)
                

                def loss_ddc(f_of_X, f_of_Y):
                    bs1=f_of_X.shape[0]
                    bs2=f_of_Y.shape[0]
                    bs=min(bs1,bs2)
                    delta = f_of_X[:bs,:] - f_of_Y[:bs,:]
                    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
                    return loss

                err_ddc=loss_ddc(s_ddc_features,t_ddc_features)

                err = self.domain_adv_coeff * (err_t_domain + err_s_domain) + self.ddc_coeff*(err_ddc) + err_s_label
                err.backward()
                clip_grad_norm_(self.model.parameters(),self.clip_value)
                if epoch>self.dual_training_epoch:
                    self.other_optimizer.step()
                    self.dual_branch_optimizer.step()
                else:
                    self.other_optimizer.step()


                cumulative_metrics["domain_loss_t_domain"]+=self.domain_adv_coeff*(err_t_domain.cpu().item()/len_dataloader)
                cumulative_metrics["domain_loss_s_domain"]+=self.domain_adv_coeff*(err_s_domain.cpu().item()/len_dataloader)
                cumulative_metrics["class_loss_s_domain"]+=err_s_label.cpu().item()/len_dataloader
                cumulative_metrics["ddc"]+=self.ddc_coeff*err_ddc.cpu().item()/len_dataloader
                cumulative_metrics["loss"]+=err.cpu().item()/len_dataloader

            print()
            print("Epoch %d"%(epoch+1))

            pprint(cumulative_metrics)
            print("On source train set: ")
            source_domain_labels=np.zeros((len(X_source),),dtype=np.int64)
            score_source_train = self.score(X_source,Y_source,"source",epoch>self.dual_training_epoch)
            for k,v in score_source_train.items():
                print(f"\t{k}:\n\t{v}")
                scores_dict[k+"_source_train"].append(v)
            print()

            print("On source val set")
            if X_source_val is not None:
                source_domain_labels=np.zeros((len(X_source_val),),dtype=np.int64)
                score_source_val = self.score(X_source_val,Y_source_val,"source",epoch>self.dual_training_epoch)
                for k,v in score_source_val.items():
                    print(f"\t{k}:\n\t{v}")
                    scores_dict[k+"_source_val"].append(v)

            print()
            print("On target set")
            target_domain_labels=np.ones((len(X_target),),dtype=np.int64)
            score_target = self.score(X_target,Y_target,"target",epoch>self.dual_training_epoch)
            for k,v in score_target.items():
                print(f"\t{k}:\n{v}")
                scores_dict[k+"_target"].append(v)
            if save:
                score_fp='{}/{}_{}-score.pkl'.format(self.model_root,self.source_dataset_name,self.target_dataset_name)

                with open(score_fp,'wb') as f:
                    pkl.dump(scores_dict,f)

                current_model_fp='{}/{}_{}-model-epoch_current.pth'.format(self.model_root,self.source_dataset_name,self.target_dataset_name)
                torch.save(self.model, current_model_fp)
                

                accu_s=score_source_train["class_accuracy"]
                if Y_target is not None:
                    accu_t=score_target["class_accuracy"]
                if (Y_target is not None and accu_t > best_accu_t) or (Y_target is None and accu_s>best_accu_s):
                    best_accu_s = accu_s
                    if Y_target is not None:
                        best_accu_t = accu_t
                    best_model_fp='{}/{}_{}-model-epoch_best.pth'.format(self.model_root,self.source_dataset_name,self.target_dataset_name)
                    torch.save(self.model, best_model_fp)
            
    def predict(self,X,domain,use_dual_branch,batch_size=None,return_score=True):
        if batch_size is None:
            batch_size=self.batch_size
        # TODO
        self.model.eval()
        X_tensor=torch.FloatTensor(X)
        dataset=torch.utils.data.TensorDataset(X_tensor)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        class_output_list=list()
        domain_output_list=list()
        device=next(iter(self.model.parameters())).device
        with torch.no_grad():
            for _,(X_batch,) in enumerate(loader):
                X_batch=X_batch.to(device)
                if use_dual_branch:
                    class_output,domain_output=self.model(X_batch,domain,use_dual_branch,alpha=self.alpha,return_ddc_features=None)
                    
                else:
                    class_output,domain_output=self.model(X_batch,None,use_dual_branch,alpha=self.alpha,return_ddc_features=None)
                if return_score:
                    class_output=class_output
                else:
                    class_output=class_output.argmax(1)
                class_output_list.append(class_output)
                domain_output_list.append(domain_output)

        class_output_tensor=torch.cat(class_output_list,0)
        domain_output_tensor=torch.cat(domain_output_list,0)
        class_output_arr=class_output_tensor.cpu().numpy()
        domain_output_arr=domain_output_tensor.cpu().numpy()
        self.model.train()
        return class_output_arr,domain_output_arr

    def score(self,X,Y,domain,use_dual_branch):
        if domain=="source":
            domain_labels=np.zeros((len(X),),dtype=np.int64)
        elif domain=="target":
            domain_labels=np.ones((len(X),),dtype=np.int64)
        if Y is not None:
            if np.min(Y)==1:
                Y=Y-1
            elif np.min(Y)!=0:
                assert False

            class_output_arr,domain_output_arr=self.predict(X,domain,use_dual_branch,return_score=True)
            class_output_tensor=torch.FloatTensor(class_output_arr)
            Y_tensor=torch.LongTensor(Y)
            domain_labels_tensor=torch.LongTensor(domain_labels)
            domain_output_tensor=torch.FloatTensor(domain_output_arr)
            class_loss=F.cross_entropy(class_output_tensor,Y_tensor,reduction="mean")
            class_output_idx_arr=class_output_arr.argmax(1)
            class_acc=np.mean(class_output_idx_arr==Y)
            domain_loss=F.cross_entropy(domain_output_tensor,domain_labels_tensor,reduction="mean")
            domain_output_idx_arr=domain_output_tensor.argmax(1).numpy()
            domain_acc=np.mean(domain_output_idx_arr==domain_labels)
            
            if class_output_arr.shape[1]==2:
                auc=roc_auc_score(Y,class_output_arr[:,1])
                return {
                    'class_accuracy':class_acc,
                    'class_loss':class_loss,
                    'domain_loss':self.domain_adv_coeff*domain_loss,
                    'domain_accuracy':domain_acc,
                    'auc':auc,
                }
            else:
                cf=confusion_matrix(Y,class_output_idx_arr)
                return {
                    'class_accuracy':class_acc,
                    'class_loss':class_loss,
                    'domain_loss':self.domain_adv_coeff*domain_loss,
                    'domain_accuracy':domain_acc,
                    'confusion_matrix':cf,
                }
            
        else:
            _,domain_output_arr=self.predict(X,domain,use_dual_branch,return_score=True)
            domain_labels_tensor=torch.LongTensor(domain_labels)
            domain_output_tensor=torch.FloatTensor(domain_output_arr)
            domain_loss=F.cross_entropy(domain_output_tensor,domain_labels_tensor,reduction="mean")
            domain_output_idx_arr=domain_output_tensor.argmax(1).numpy()
            domain_acc=np.mean(domain_output_idx_arr==domain_labels)
            return {
                    'domain_loss':self.domain_adv_coeff*domain_loss,
                    'domain_accuracy':domain_acc,
            }
    
