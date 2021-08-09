import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from model.bert import BertPreTrainedModel, BertModel
from model.tamm import TaMM

class ReTamm(BertPreTrainedModel):
    def __init__(self, config):
        super(ReTamm, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*4, config.num_labels)
        self.ensemble_linear = nn.Linear(1, 2)
        self.kvmn = TaMM(config.hidden_size, config.key_size, config.val_size)
        self.apply(self.init_bert_weights)

    def valid_filter(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype,
                                   device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        return valid_output

    def extract_entity(self, sequence, e_mask):
        entity_output = sequence * torch.stack([e_mask] * sequence.shape[-1], 2) + torch.stack(
            [(1.0 - e_mask) * -1000.0] * sequence.shape[-1], 2)
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(sequence)

    def get_entity_from_kvmn(self, sequence_output, dep_key_list, dep_adj_matrix, dep_type_matrix, e1_mask, e2_mask):
        kvmn_output = self.kvmn(sequence_output, dep_key_list, dep_adj_matrix, dep_type_matrix)
        kv_e1_h = self.extract_entity(kvmn_output, e1_mask)
        kv_e2_h = self.extract_entity(kvmn_output, e2_mask)
        return kv_e1_h, kv_e2_h

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, e1_mask=None, e2_mask=None,
                dep_key_list=None, dep_order_dep_adj_matrix=None, dep_order_dep_type_matrix=None,
                dep_path_dep_adj_matrix=None, dep_path_dep_type_matrix=None, valid_ids=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        if valid_ids is not None:
            valid_sequence_output = self.valid_filter(sequence_output, valid_ids)
        else:
            valid_sequence_output = sequence_output

        kv_in_entity_e1_h, kv_in_entity_e2_h = self.get_entity_from_kvmn(valid_sequence_output, dep_key_list,
                                                                         dep_order_dep_adj_matrix,
                                                                         dep_order_dep_type_matrix, e1_mask, e2_mask)
        kv_cross_entity_e1_h, kv_cross_entity_e2_h = self.get_entity_from_kvmn(valid_sequence_output, dep_key_list,
                                                                               dep_path_dep_adj_matrix,
                                                                               dep_path_dep_type_matrix, e1_mask, e2_mask)

        cls_output = torch.cat([kv_in_entity_e1_h, kv_in_entity_e2_h, kv_cross_entity_e1_h, kv_cross_entity_e2_h], dim=-1)
        cls_output = self.dropout(cls_output)

        logits = self.classifier(cls_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            return logits