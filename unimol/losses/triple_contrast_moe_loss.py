# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json

import math
import torch
import torch.nn.functional as F
import pandas as pd
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import warnings
from sklearn.metrics import top_k_accuracy_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC
import random
import scipy.stats as stats
from scipy import stats
import debugpy
from torch_scatter import scatter_mean, scatter_add



def calculate_bedroc(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """

    # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    # print(scores.shape, y_true.shape)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:, 0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    return bedroc


@register_loss("triple_moe_contrast")
class Triple_MoE_RSLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.loss = torch.nn.CrossEntropyLoss()
        #self.args = task.args

    def forward(self, net_output, sample, logit_scale, reduce=True, fix_encoder=False):
        """
        Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.training:
            pocket_emb, seq_emb = net_output[0]
            mol_emb = net_output[1][0]

            logit_output_pkt = torch.matmul(pocket_emb, torch.transpose(mol_emb, 0, 1))
            logit_output_pkt = logit_output_pkt * logit_scale.exp()#.detach()
            loss_dict_pkt = self.compute_loss(logit_output_pkt, sample, reduce=reduce)
            
            if self.args.use_sequence:
                logit_output_seq = torch.matmul(seq_emb, torch.transpose(mol_emb, 0, 1))
                logit_output_seq = logit_output_seq * logit_scale.exp()#.detach()
                loss_dict_seq = self.compute_loss(logit_output_seq, sample, reduce=reduce)

                loss_dict_accum = {}
                for k in loss_dict_pkt.keys():
                    pkt_loss = loss_dict_pkt[k]
                    seq_loss = loss_dict_seq[k]

                    inv_pkt = 1.0 / (pkt_loss.detach() + 1e-6)
                    inv_seq = 1.0 / (seq_loss.detach() + 1e-6)
                    w_pkt = inv_pkt / (inv_pkt + inv_seq)
                    w_seq = inv_seq / (inv_pkt + inv_seq)

                    loss_dict_accum[f'pkt_{k}'] = pkt_loss
                    loss_dict_accum[f'seq_{k}'] = seq_loss

                    loss_dict_accum[k] = w_pkt * pkt_loss + w_seq * seq_loss
            else:
                loss_dict_accum = loss_dict_pkt
            if self.args.l2_loss:
                l2_loss = self.l2_loss(
                    net_output[2], net_output[3], net_output[4], sample['batch_list'],
                    )
                loss_dict_accum['loss_l2'] = l2_loss
                loss_dict_accum['loss'] = loss_dict_accum['loss'] + l2_loss
            aux_loss, n_aux_loss = self.moe_aux_loss(net_output)
            if n_aux_loss > 0:
                loss_dict_accum['loss'] = loss_dict_accum['loss'] + aux_loss
                loss_dict_accum['loss_aux'] = aux_loss
        else:
            pocket_emb = net_output[0][0]
            mol_emb = net_output[1][0]
            logit_output = torch.matmul(pocket_emb, torch.transpose(mol_emb, 0, 1))
            logit_output = logit_output * logit_scale[0].exp().detach()
            loss_dict_accum = {"loss": torch.tensor(0., device=logit_output.device)}

        if not self.training:
            # For training
            if self.args.valid_set in ["FEP", "TIME", "TYK2", "OOD", "DEMO"]:
                sample_size = logit_output.size(0)
                logging_output = {
                    "loss": loss_dict_accum["loss"].data,
                    "logit_output": logit_output,
                    "act_list": sample["act_list"],
                    "batch_list": sample["batch_list"],
                    "smi_name": sample["lig"]["smi_name"],
                    "sample_size": sample_size,
                    "assay_id_list": sample["assay_id_list"],
                    "bsz": logit_output.size(0),
                    "scale": logit_scale[0].data
                }
            else:
                sample_size = logit_output.size(0)
                print('logit_output: ', logit_output.shape)
                targets = torch.arange(sample_size, dtype=torch.long).cuda()
                assert logit_output.size(1) == sample_size
                logit_output = logit_output[:, :sample_size]
                probs = F.softmax(logit_output.float(), dim=-1).view(
                    -1, logit_output.size(-1)
                )
                logging_output = {
                    "loss": loss_dict_accum["loss"].data,
                    "prob": probs.data,
                    "target": targets,
                    "smi_name": sample["lig"]["smi_name"],
                    "sample_size": sample_size,
                    "bsz": logit_output.size(0),
                    "scale": logit_scale.data
                }
        else:
            sample_size = pocket_emb.size(0)
            logging_output = {}
            for k,v in loss_dict_accum.items():
                logging_output[k] = v.data
            logging_output.update({
                "sample_size": sample_size,
                "bsz": pocket_emb.size(0),
                "scale": logit_scale.data
                })
        return loss_dict_accum["loss"], sample_size, logging_output

    def moe_aux_loss(self, net_output):
        mol_aux_loss = net_output[-2]
        pkt_aux_loss = net_output[-1]
        aux_loss, n_aux_loss = 0, 0
        if mol_aux_loss is not None:
            aux_loss += mol_aux_loss
            n_aux_loss += 1
        if pkt_aux_loss is not None:
            aux_loss += pkt_aux_loss
            n_aux_loss += 1
        if n_aux_loss > 0:
            aux_loss = aux_loss / n_aux_loss
        return aux_loss, n_aux_loss

    def l2_loss(self, lig_embs, target_embs, seq_embs, batch_list):
        mol_rep, mol_pred = lig_embs
        pkt_rep, pkt_pred = target_embs
        seq_rep, seq_pred = seq_embs

        batch, lig_num_per_pkt = [], []
        for ix, i in enumerate(batch_list):
            batch += [ix] * (i[1] - i[0])
            lig_num_per_pkt.append(i[1] - i[0])

        batch = torch.tensor(batch).long().to(pkt_pred.device)
        lig_num_per_pkt = torch.tensor(lig_num_per_pkt, dtype=pkt_pred.dtype).to(pkt_pred.device)
        n_l2_loss = 1
        pkt_l2_loss = F.smooth_l1_loss(pkt_pred, pkt_rep[batch], reduction='none')
        pkt_l2_loss = pkt_l2_loss / torch.sqrt(lig_num_per_pkt)[batch].unsqueeze(1)
        pkt_l2_loss = pkt_l2_loss.mean(-1).sum()

        if self.args.l2_pkt2mol:
            mol_l2_loss = F.smooth_l1_loss(mol_pred[batch], mol_rep, reduction='none')
            mol_l2_loss = mol_l2_loss / torch.sqrt(lig_num_per_pkt)[batch].unsqueeze(1)
            mol_l2_loss = mol_l2_loss.mean(-1).sum()
            pkt_l2_loss += mol_l2_loss
            n_l2_loss += 1
        
        if self.args.l2_sequence:
            seq_l2_loss = F.smooth_l1_loss(seq_pred, seq_rep[batch], reduction='none')
            seq_l2_loss = seq_l2_loss / torch.sqrt(lig_num_per_pkt)[batch].unsqueeze(1)
            seq_l2_loss = seq_l2_loss.mean(-1).sum()
            pkt_l2_loss += seq_l2_loss
            n_l2_loss += 1
            
        l2_loss = pkt_l2_loss / n_l2_loss
        return l2_loss

    def compute_loss(self, net_output, sample, reduce=True):
        batch_list = sample["batch_list"]
        act_list = sample["act_list"]
        smi_list = sample["lig"]["smi_name"]
        uniprotid_list = sample["uniprot_list"]
        net_output = net_output.float()
        num_pocket = net_output.shape[0]
        num_lig = net_output.shape[1]
        idx2pocket = []

        uniprotid_mask = torch.zeros_like(net_output)
        mol_mask = torch.zeros_like(net_output)
        for i in range(num_pocket):
            range_i = batch_list[i]
            idx2pocket += [i] * (range_i[1] - range_i[0])
            smi_pocket_i = smi_list[range_i[0]: range_i[1]]
            for j in range(num_pocket):
                if j == i:
                    continue
                range_j = batch_list[j]

                # mols of other pocket with the same uniprot id should be ignored
                if uniprotid_list[i] == uniprotid_list[j]:
                    uniprotid_mask[i, range_j[0]:range_j[1]] = -1e9

                # mols of other pocket with the same smi should be ignored
                for k in range(range_j[0], range_j[1]):
                    if smi_list[k] in smi_pocket_i:
                        mol_mask[i, k] = -1e9

        net_output = net_output + uniprotid_mask + mol_mask

        def pcc(x, y):
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        # pocket retrieve mol
        loss_mol = []
        loss_rank = []
        loss_corr = []
        for i in range(num_pocket):
            range_i = batch_list[i]
            act_list_i = act_list[i]
            for k in range(range_i[0], range_i[1]):
                mask = torch.zeros_like(net_output[i])
                mask[range_i[0]: range_i[1]] = -1e9
                mask[k] = 0.
                lprobs_mol = F.log_softmax(mask + net_output[i], dim=-1)
                loss_tmp = F.nll_loss(
                    lprobs_mol,
                    torch.tensor(k).to(lprobs_mol.device),
                    reduction="sum" if reduce else "none",
                )
                if range_i[1] - range_i[0] > 1 and act_list_i[k - range_i[0]] < 5:
                    continue
                loss_mol.append(loss_tmp / math.sqrt(range_i[1] - range_i[0]))

            if self.args.rank_loss:
                if range_i[1] - range_i[0] > 2:
                    output_i = net_output[i, range_i[0]:range_i[1]]
                    act_list_i = act_list[i]
                    for k in range(range_i[1] - range_i[0] - 1):
                        mask = torch.zeros_like(output_i)
                        # mask[:k] = -1e9
                        for idx in range(0, range_i[1] - range_i[0]):
                            if idx == k:
                                continue
                            if act_list_i[k] - math.log10(3) <= act_list_i[idx]:  # three times
                                mask[idx] = -1e9
                        lprobs_mol = F.log_softmax(mask + output_i, dim=-1)
                        loss_tmp = F.nll_loss(
                            lprobs_mol,
                            torch.tensor(k).to(lprobs_mol.device),
                            reduction="sum" if reduce else "none",
                        )
                        loss_rank.append(loss_tmp / (math.log(k + 2) * math.sqrt(range_i[1] - range_i[0])))

                    corr = pcc(output_i, torch.tensor(act_list_i).to(output_i.device))
                    if not torch.isnan(corr):
                        loss_corr.append(1. - corr)

        loss_mol = torch.stack(loss_mol).sum()

        lprobs_pocket = F.log_softmax(torch.transpose(net_output, 0, 1), dim=-1)
        lprobs_pocket = lprobs_pocket.view(-1, lprobs_pocket.size(-1))
        targets = torch.tensor(idx2pocket, dtype=torch.long).view(-1).to(lprobs_pocket.device)
        loss_pocket_ = F.nll_loss(
            lprobs_pocket,
            targets,
            reduction="none",
        )
        loss_pocket = []
        for i in range(num_pocket):
            range_i = batch_list[i]
            if range_i[1] - range_i[0] > 0:
                loss_pocket.append(loss_pocket_[range_i[0]:range_i[1]].sum() / math.sqrt(range_i[1] - range_i[0]))
        loss_pocket = torch.stack(loss_pocket).sum()
        loss = self.args.contras_weight * (loss_pocket + loss_mol)
        if self.args.rank_loss:
            if self.args.few_shot:
                loss_rank = loss_corr
                self.args.contras_weight = 0.

            if len(loss_rank) > 0:
                loss_rank = torch.stack(loss_rank).sum()
                loss = loss + self.args.rank_weight * loss_rank
            else:
                loss_rank = 0. * loss

            return {"loss": loss,
                    "loss_pocket": loss_pocket,
                    "loss_mol": loss_mol,
                    "loss_rank": loss_rank}
        else:
            return {"loss": loss,
                    "loss_pocket": loss_pocket,
                    "loss_mol": loss_mol}

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid", args=None) -> None:
        """Aggregate logging outputs from data parallel training."""
        metrics.log_scalar("scale", logging_outputs[0].get("scale"), round=3)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        valid_set = args.valid_set
        split_method = args.split_method
        if "train" in split:
            #print(logging_outputs)
            for key in ["loss_mol", "loss_pocket", "loss_l2", "seq_loss", "pkt_loss"]:
                loss_sum = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key, loss_sum / sample_size, sample_size, round=3
                )
            if "loss_rank" in logging_outputs[0]:
                loss_rank = sum(log.get("loss_rank", 0) for log in logging_outputs)
                metrics.log_scalar(
                    "loss_rank", loss_rank / sample_size, sample_size, round=3
                )
            if "loss_aux" in logging_outputs[0]:
                loss_aux = sum(log.get("loss_aux", 0) for log in logging_outputs)
                metrics.log_scalar(
                    "loss_aux", loss_aux, sample_size, round=3
                )
        elif valid_set in ["FEP", "TIME", "TYK2", "OOD", "DEMO"]:
            corrs = []
            pearsons = []
            r2s = []
            res_dict = {}
            info_dict = {}
            for log in logging_outputs:
                logit_output = log["logit_output"].detach().cpu().numpy()
                true_act = log["act_list"]
                lig_smi = log["smi_name"]
                for i, (assay_id, span, acts) in enumerate(zip(log["assay_id_list"], log["batch_list"], true_act)):
                    acts = np.array(acts)
                    if len(acts) >= 3:
                        pred_score = logit_output[i, span[0]:span[1]]
                        corr = stats.spearmanr(acts, pred_score).statistic
                        pearson = stats.pearsonr(acts, pred_score).statistic
                        if math.isnan(corr):
                            corr = 0.
                        if math.isnan(pearson):
                            pearson = 0.
                        assay_smi = lig_smi[span[0]:span[1]]
                        res_dict[assay_id] = {
                            "assay_id": assay_id,
                            "pred": [round(x, 3) for x in pred_score.tolist()],
                            "exp": [round(x, 3) for x in acts.tolist()],
                            "spearmanr": corr,
                            "pearson": pearson
                        }
                        info_dict[assay_id] = {
                            "assay_id": assay_id,
                            "smiles": assay_smi,
                        }
                        corrs.append(corr)
                        pearsons.append(pearson)
                        r2s.append(max(pearson, 0) ** 2)

            metrics.log_scalar(f"{split}_mean_corr", np.mean(corrs), sample_size, round=3)
            metrics.log_scalar(f"{split}_mean_pearson", np.mean(pearsons), sample_size, round=3)
            metrics.log_scalar(f"{split}_mean_r2", np.mean(r2s), sample_size, round=3)
            sup_num = float(args.sup_num)
            if args.sup_num > 1:
                sup_num = int(args.sup_num)

            import os
            rank = int(os.environ["LOCAL_RANK"])
            if rank == 0 and args.few_shot:
                if args.results_path.endswith(".jsonl"):
                    write_file = args.results_path
                else:
                    write_file = f"{args.results_path}/{split_method}_{args.seed}_sup{sup_num}.jsonl"
                    if args.active_learning_resfile != "":
                        write_file = f"{args.results_path}/{args.active_learning_resfile}"
                import os
                if not os.path.exists(write_file):
                    with open(write_file, "a") as f:
                        f.write(json.dumps(info_dict) + "\n")
                with open(write_file, "a") as f:
                    f.write(json.dumps(res_dict) + "\n")
                print(f"saving to {write_file}")

        else:
            acc_sum = sum(sum(log.get("prob").argmax(dim=-1) == log.get("target")) for log in logging_outputs)

            prob_list = []
            if len(logging_outputs) == 1:
                prob_list.append(logging_outputs[0].get("prob"))
            else:
                for i in range(len(logging_outputs) - 1):
                    prob_list.append(logging_outputs[i].get("prob"))
            probs = torch.cat(prob_list, dim=0)
            #print('probs: ', probs.shape)  #  [240, 48]
            metrics.log_scalar(f"{split}_acc", acc_sum / sample_size, sample_size, round=3)
            metrics.log_scalar("valid_neg_loss", -loss_sum / sample_size / math.log(2), sample_size, round=3)
            targets = torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)

            targets = targets[:len(probs)]
            bedroc_list = []
            auc_list = []
            for i in range(len(probs)):
                prob = probs[i]
                target = targets[i]
                label = torch.zeros_like(prob)
                label[target] = 1.0
                #print('label: ', label.shape, label)  #[48]
                cur_auc = roc_auc_score(label.cpu(), prob.cpu())
                auc_list.append(cur_auc)
                bedroc = calculate_bedroc(label.cpu(), prob.cpu(), 80.5)
                bedroc_list.append(bedroc)
            bedroc = np.mean(bedroc_list)
            auc = np.mean(auc_list)

            top_k_acc = top_k_accuracy_score(targets.cpu(), probs.cpu(), k=3, normalize=True)
            metrics.log_scalar(f"{split}_auc", auc, sample_size, round=3)
            metrics.log_scalar(f"{split}_bedroc", bedroc, sample_size, round=3)
            metrics.log_scalar(f"{split}_top3_acc", top_k_acc, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
    