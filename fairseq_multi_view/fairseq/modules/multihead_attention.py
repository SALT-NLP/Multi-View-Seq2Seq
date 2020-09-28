# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor, nn
from torch.nn import Parameter
from fairseq.incremental_decoding_utils import with_incremental_state


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        self.enable_torch_version = False
        #if hasattr(F, "multi_head_attention_forward"):
        #    self.enable_torch_version = True
        #else:
        #    self.enable_torch_version = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,

        key2: Optional[Tensor] = None,
        value2: Optional[Tensor] = None,
        balance_weight = None,
        key_padding_mask2 = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if (
            self.enable_torch_version
            and not self.onnx_trace
            and incremental_state is None
            and not static_kv
        ):
            assert key is not None and value is not None
            #print("here")
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

        if incremental_state is not None:
            #print("incremental_state not None")
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None

            if saved_state is not None and "prev_key2" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key2 = value2 = None

        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
            k2 = None
            v2 = None
        elif self.encoder_decoder_attention:
            
            #print('key', key)

            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
                k2 = None
                v2 = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

                #print("here!!!")
                k2 = None
                v2 = None
                if key2 is not None:
                    k2 = self.k_proj(key2)
                    v2 = self.v_proj(key2)

            #print('k', k)
            #print('v', v)


        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
            k2 = None
            v2 = None
        
        q *= self.scaling

        

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])

            if key2 is not None:
                k2 = torch.cat([k2, self.bias_k.repeat(1, bsz, 1)])
                v2 = torch.cat([v2, self.bias_v.repeat(1, bsz, 1)])


            if attn_mask is not None:
                print("Attn mask not None")
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )
            if key_padding_mask2 is not None:
                key_padding_mask2 = torch.cat(
                    [
                        key_padding_mask2,
                        key_padding_mask2.new_zeros(key_padding_mask2.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if k2 is not None:
            k2 = (
                k2.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        

        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if v2 is not None:
            v2 = (
                v2.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            #print("Here saved state is not None!!!")

            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)

            
            if "prev_key2" in saved_state:
                _prev_key2 = saved_state["prev_key2"]
                assert _prev_key2 is not None
                prev_key2 = _prev_key2.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k2 = prev_key2
                else:
                    assert k2 is not None
                    k2 = torch.cat([prev_key2, k2], dim=1)

            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)

            if "prev_value2" in saved_state:
                #print("in")
                _prev_value2 = saved_state["prev_value2"]
                assert _prev_value2 is not None
                prev_value2 = _prev_value2.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v2 = prev_value2
                else:
                    assert v2 is not None
                    v2 = torch.cat([prev_value2, v2], dim=1)


            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]

            prev_key_padding_mask2: Optional[Tensor] = None
            if "prev_key_padding_mask2" in saved_state:
                prev_key_padding_mask2 = saved_state["prev_key_padding_mask2"]
            

            assert k is not None and v is not None
            
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            if key_padding_mask2 is not None:
                key_padding_mask2 = MultiheadAttention._append_prev_key_padding_mask(
                    key_padding_mask=key_padding_mask2,
                    prev_key_padding_mask=prev_key_padding_mask2,
                    batch_size=bsz,
                    src_len=k2.size(1),
                    static_kv=static_kv,
                )



            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask

            if k2 is not None:
                saved_state["prev_key2"] = k2.view(bsz, self.num_heads, -1, self.head_dim)

            if v2 is not None:
                saved_state["prev_value2"] = v2.view(bsz, self.num_heads, -1, self.head_dim)
            
            if key_padding_mask2 is not None:
                saved_state["prev_key_padding_mask2"] = key_padding_mask2

            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        
        assert k is not None
        src_len = k.size(1)

        if k2 is not None:
            src_len2 = k2.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask2 is not None and key_padding_mask2.dim() == 0:
            key_padding_mask2 = None

        #print(key_padding_mask.shape)
        #print(key_padding_mask2.shape)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if key_padding_mask2 is not None:
            assert key_padding_mask2.size(0) == bsz
            assert key_padding_mask2.size(1) == src_len2

        if self.add_zero_attn:
            #print("add zero attn")
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        #print("not add zero attn")
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)


        if k2 is not None:
            attn_weights2 = torch.bmm(q, k2.transpose(1, 2))
            attn_weights2 = MultiheadAttention.apply_sparse_mask(attn_weights2, tgt_len, src_len2, bsz)

        
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask
            if k2 is not None:
                attn_weights2 += attn_mask



        

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask2 is not None:
            # don't attend to padding symbols
            #print("padding attn_weights2")
            attn_weights2 = attn_weights2.view(bsz, self.num_heads, tgt_len, src_len2)
            attn_weights2 = attn_weights2.masked_fill(
                key_padding_mask2.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights2 = attn_weights2.view(bsz * self.num_heads, tgt_len, src_len2)

        if before_softmax:
            
            return attn_weights, v


        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )

        if key_padding_mask2 is not None:
            attn_weights_float2 = utils.softmax(
            attn_weights2, dim=-1, onnx_trace=self.onnx_trace
            )
            attn_weights2 = attn_weights_float2.type_as(attn_weights2)
            attn_probs2 = F.dropout(
                attn_weights_float2.type_as(attn_weights2),
                p=self.dropout,
                training=self.training,
            )
        else:
            attn_weights_float2 = None
            attn_probs2  = None

        assert v is not None




        attn = torch.bmm(attn_probs, v)

        #print('embed_dim', embed_dim)
        #print("sec_len", src_len)
        #print("target_len", tgt_len)
        #print("attn_probs", attn_probs.shape)
        #print("v", v.shape)
        #print("num heads", self.num_heads)

        if v2 is not None:
            #print('not None')
            attn2 = torch.bmm(attn_probs2, v2)

        '''
        if balance_weight is not None:

            #print('balance_weight', balance_weight.shape)

            balance_weight = balance_weight.repeat([self.num_heads, 1])

            balance_weight1 = balance_weight[:,0].view([balance_weight.shape[0], 1, 1])

            #balance_weight2 = balance_weight[:,1].view([balance_weight.shape[0], 1, 1])
            
            balance_weight1 = balance_weight1.repeat([1, tgt_len, self.head_dim])
            #balance_weight2 = balance_weight2.repeat([1, tgt_len, self.head_dim])

            
            #attn = balance_weight1 * attn + balance_weight2 * attn2

            #attn = attn 

            #attn = 0.6 * attn + 0.4 * attn2
            attn = 0.8 * attn + 0.2 * attn2
            #print('balance_weight', balance_weight.shape)
            #print('balance_weight1', balance_weight1.shape)
            #print('attn shape', attn.shape)
            #print('attn weights float', attn_weights_float.shape)
            # 64 * 60 * 192

            #attn = balance_weight1 * attn + (1 - balance_weight1) * attn2

        '''

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        
        if self.onnx_trace and attn.size(1) == 1:
            print("here onnx_trace")
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
            if v2 is not None:
                attn2 = attn2.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            #print('here')
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            if v2 is not None:
                attn2 = attn2.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        
        
        attn = self.out_proj(attn)

        if v2 is not None:
            attn2 = self.out_proj(attn2)
        
        #if v2 is not None:
        #    print('attn', attn)
        #    print('attn2', attn2)

        #if v2 is not None:
            #attn =  1.0* attn  + 0 * attn2
        #    attn = 0 * attn + 1 * attn2
        #print(attn)
        
        if balance_weight is not None:
            #print(balance_weight.shape)
            
            balance_weight_temp = torch.cat(tgt_len * [balance_weight.unsqueeze(0)], dim = 0).unsqueeze(-1)
            #print(balance_weight_temp.shape)
            temp_attn = torch.cat([attn.unsqueeze(2), attn2.unsqueeze(2)], dim = 2)
            attn = temp_attn.mul(balance_weight_temp)
            attn = torch.sum(attn, dim = 2)

            #print(balance_weight)
            
            #print(attn.shape)

        #attn = attn2
        
        attn_weights: Optional[Tensor] = None
        attn_weights2 = None

        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)

            if attn_weights_float2 is not None:
                attn_weights2 = attn_weights_float2.view(
                    bsz, self.num_heads, tgt_len, src_len2
                ).transpose(1, 0)

            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
                if attn_weights2 is not None:   
                    attn_weights2 = attn_weights2.mean(dim = 0)
        


        #attn_weights = None
        #attn_weights2 = None

        if attn_weights2 is not None and need_weights:
            return attn,  attn_weights,  attn_weights2
        else:
            return attn, attn_weights, None

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:

            filler = torch.zeros(batch_size, src_len - prev_key_padding_mask.size(1))
            if prev_key_padding_mask.is_cuda:
                filler = filler.cuda()
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - key_padding_mask.size(1))
            if key_padding_mask.is_cuda:
                filler = filler.cuda()
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value



