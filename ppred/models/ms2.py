"""Checkpoint-compatible graph and transformer model for MS2 intensities."""
import numpy as np
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GlobalAttention, Set2Set

from .gnn import GNN

max_instrument_num = 8
composition_x_feature_size = 118


class Composition_Embedding_FixFirstK(torch.nn.Module):
    """
    Encodes the modification vector in a single layer feed forward network, but not transforming the first k features
    """
    def __init__(self,
        out_features,
    ):
        super().__init__()
        self.k = 6
        self.nn = torch.nn.Linear(
            composition_x_feature_size-self.k, out_features-self.k,
            bias=False
        )

    def forward(self,
        composition_x,
    ):
        self.nn.to(composition_x.device)
        nn_output = self.nn(composition_x[:,:,self.k:])
        return torch.cat((composition_x[:,:,:self.k], nn_output), 2)


class PositionalEncoding(torch.nn.Module):
    """
    transform sequence input into a positional representation
    """
    def __init__(self, out_features=128, max_len = 200):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0, out_features, 2
            ) * (-np.log(max_len) / out_features)
        )
        pe = torch.zeros(1, max_len, out_features)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:,:x.size(1),:]


class Meta_Embedding(torch.nn.Module):
    """Encodes Charge state, Normalized Collision Energy (NCE) and Instrument for a given spectrum
    into a 'meta' single layer network
    """
    def __init__(self,
        out_features,
    ):
        super().__init__()
        self.nn = torch.nn.Linear(
            max_instrument_num+1, out_features-1
        )

    def forward(self,
        charges, NCEs, instrument_indices,
    ):
        inst_x = torch.nn.functional.one_hot(
            instrument_indices, max_instrument_num
        )
        meta_x = self.nn(torch.cat((inst_x, NCEs), 1))
        meta_x = torch.cat((meta_x, charges), 1)
        return meta_x

class _Pseudo_Bert_Config:
    def __init__(self,
        hidden_dim=256,
        intermediate_size=1024,
        num_attention_heads=8,
        num_bert_layers=4,
        dropout=0.1,
        output_attentions=False,
    ):
        self.add_cross_attention = False
        self.chunk_size_feed_forward = 0
        self.is_decoder = False
        self.seq_len_dim = 1
        self.training = False
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = dropout
        self.attention_probs_dropout_prob = dropout
        self.hidden_size = hidden_dim
        self.initializer_range = 0.02
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = 1e-8
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_bert_layers
        self.output_attentions = output_attentions

def invert_attention_mask(
    encoder_attention_mask:torch.Tensor,
    dtype=torch.float32
)->torch.FloatTensor:
    """
    See `invert_attention_mask()` in https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L737.
    Invert an attention mask (e.g., switches 0. and 1.).
    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.
    Returns:
        `torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(dtype).min

    return encoder_extended_attention_mask

class Hidden_HFace_Transformer(torch.nn.Module):
    """
    Transformer NN based on HuggingFace's BertEncoder class
    """
    def __init__(self,
        hidden_dim, hidden_expand=4,
        nheads=8, nlayers=4, dropout=0.1,
        output_attentions=False,
    ):
        super().__init__()
        self.config = _Pseudo_Bert_Config(
            hidden_dim=hidden_dim,
            intermediate_size=hidden_dim*hidden_expand,
            num_attention_heads=nheads,
            num_bert_layers=nlayers,
            dropout=dropout,
            output_attentions=False
        )
        self.output_attentions = output_attentions
        self.bert = BertEncoder(self.config)
    def forward(self, x:torch.Tensor,
        attention_mask:torch.Tensor=None,
    )->tuple:
        """
        Parameters
        ----------
        x : torch.Tensor
            shape = (batch, seq_len, dim)
        attention_mask : torch.Tensor
            shape = (batch, seq_len), [0,1] tensor , 1=enable

        Returns
        -------
        (Tensor, [Tensor])
            out[0] is the hidden layer output,
            and out[1] is the output attention
            if self.output_attentions==True
        """
        if attention_mask is not None:
            attention_mask = invert_attention_mask(
                attention_mask, dtype=x.dtype
            )
        return self.bert(
            x,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            return_dict=False
        )

class Decoder_Linear(torch.nn.Module):
    """
    Decode w linear NN
    """
    def __init__(self, in_features, out_features):
        super().__init__()

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.PReLU(),
            torch.nn.Linear(64, out_features),
        )

    def forward(self, x):
        return self.nn(x)

class ModelMS2Bert(torch.nn.Module):
    """Using HuggingFace's BertEncoder for MS2 prediction"""
    def __init__(self,
        num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin", device=None,

        num_frag_types=8,
        num_modloss_types=0,
        mask_modloss=True,
        dropout=0.1,
        nlayers=4,
        hidden=256,
        output_attentions=False,
        **kwargs,
    ):
        super().__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.device = device
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(self.num_layer, self.emb_dim, self.device, self.JK, self.drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(
                    (self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1)
                                    * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)

        # Retain legacy graph-head parameters for strict checkpoint compatibility.
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(
                self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_embedding = nn.ModuleList([
                torch.nn.Linear(3 * self.mult * self.emb_dim, self.mult * self.emb_dim),
                torch.nn.Linear(self.mult * self.emb_dim, self.mult * self.emb_dim),
                torch.nn.Linear(self.mult * self.emb_dim, self.mult * self.emb_dim),
                torch.nn.Linear(self.mult * self.emb_dim, self.mult * self.emb_dim)
            ])
            self.graph_pred_linear = torch.nn.Linear(
                self.mult * self.emb_dim, self.num_tasks)
        self.graph_pred_activation = torch.nn.ReLU()

        self.fp_fc1=nn.Linear(1489, 512)
        self.fp_act_func = nn.ReLU()
        self.fp_fc2 = nn.Linear(512, 64)
        self.fp_dropout = nn.Dropout(p=0.0)

        self.dropout = torch.nn.Dropout(dropout)

        self._num_modloss_types = num_modloss_types
        self._num_non_modloss = num_frag_types-num_modloss_types
        self._mask_modloss = mask_modloss
        if num_modloss_types == 0:
            self._mask_modloss = True

        meta_dim = 8
        mod_hidden = 248
        self.composition_nn = Composition_Embedding_FixFirstK(mod_hidden)

        self.pos_encoder = PositionalEncoding(
            hidden-meta_dim, max_len=200
        )

        self.meta_nn = Meta_Embedding(meta_dim)  # Retained checkpoint parameters.

        self.max_instrument_num = 8
        self.instr_nn = torch.nn.Linear(
            self.max_instrument_num+2, meta_dim-1
        )

        self._output_attentions = output_attentions

        self.hidden_nn = Hidden_HFace_Transformer(
            hidden+128, nlayers=nlayers, dropout=dropout,
            output_attentions=output_attentions
        )

        self.output_nn = Decoder_Linear(
            hidden+128,
            self._num_non_modloss,
        )

        if num_modloss_types > 0:
            # for transfer learning of modloss frags
            self.modloss_nn = torch.nn.ModuleList([
                Hidden_HFace_Transformer(
                    hidden+128, nlayers=1, dropout=dropout,
                    output_attentions=output_attentions
                ),
                Decoder_Linear(
                    hidden+128, num_modloss_types,
                ),
            ])
        else:
            self.modloss_nn = None


    @property
    def output_attentions(self):
        return self._output_attentions

    @output_attentions.setter
    def output_attentions(self, val:bool):
        self._output_attentions = val
        self.hidden_nn.output_attentions = val
        self.modloss_nn[0].output_attentions = val

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        fp = data.fp
        fp = fp.reshape(-1, 1489)
        fpn_out = self.fp_fc1(fp.float())
        fpn_out = self.fp_dropout(fpn_out)
        fpn_out = self.fp_act_func(fpn_out)
        fpn_out = self.fp_fc2(fpn_out)

        node_representation = self.gnn(x, edge_index, edge_attr)
        node_pool = self.pool(node_representation, batch)

        nce_x = data.nce
        nce_x = nce_x.unsqueeze(1)
        charge_x = data.charge
        charge_x = charge_x.unsqueeze(1)
        instrument_indices_x = data.inst_indices
        activation_x = data.actication.unsqueeze(1)
        batch_size = nce_x.shape[0]
        composition_x = data.element_composition_array.reshape(batch_size, -1, 118)

        composition_x1 = self.composition_nn(composition_x.float())
        self.pos_encoder.to(composition_x1.device)
        composition_x2 = self.pos_encoder(composition_x1)

        instrument_indices_x = instrument_indices_x.to(torch.long)
        inst_x = torch.nn.functional.one_hot(
            instrument_indices_x, self.max_instrument_num
        )
        inst_x = torch.cat((inst_x, nce_x), 1)
        inst_x = torch.cat((inst_x, activation_x), 1)

        inst_x1 = self.instr_nn(inst_x.float())
        inst_x2 = torch.cat((inst_x1, charge_x), 1)

        inst_x2 = torch.cat((inst_x2, fpn_out), 1)
        inst_x2 = torch.cat((inst_x2, node_pool), 1)

        inst_x2 = inst_x2.unsqueeze(1).repeat(1, composition_x2.size(1), 1)

        input_x = torch.cat((composition_x2, inst_x2), 2).float()
        hidden_x = self.hidden_nn(input_x)
        if self.output_attentions:
            self.attentions = hidden_x[1]
        else:
            self.attentions = None
        hidden_x = self.dropout(hidden_x[0]+input_x*0.2)
        out_x = self.output_nn(
            hidden_x
        )

        self.modloss_attentions = None
        if self._num_modloss_types > 0:
            if self._mask_modloss:
                out_x = torch.cat((out_x, torch.zeros(
                    *out_x.size()[:2],self._num_modloss_types,
                    device=input_x.device
                )), 2)
            else:
                modloss_x = self.modloss_nn[0](
                    input_x
                )
                if self.output_attentions:
                    self.modloss_attentions = modloss_x[-1]
                modloss_x = modloss_x[0] + hidden_x
                modloss_x = self.modloss_nn[-1](
                    modloss_x
                )
                out_x = torch.cat((
                    out_x, modloss_x
                ),2)
        return out_x[:,1:,:]

