# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""

import logging
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
import pickle

from torch.nn import TransformerEncoderLayer as tel
from torch.nn import TransformerEncoder as te

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .hub_interface import BARTHubInterface

import torch.nn.functional as F

from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


logger = logging.getLogger(__name__)




@register_model('bart')
class BARTModel(TransformerModel):

    @classmethod
    def hub_models(cls):
        return {
            'bart.large': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz',
            'bart.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz',
            'bart.large.cnn': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz',
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

        self.T = args.T

        #TODO: 

        self.section_positions = PositionalEmbedding(
                    args.max_source_positions,
                    1024,
                    0,
                    learned=args.encoder_learned_pos)
        self.section_layernorm_embedding = LayerNorm(1024)
            
        #self.section = te(tel(d_model=1024, nhead=1), num_layers=1, norm = LayerNorm(1024))
        self.section = nn.LSTM(input_size = 1024, hidden_size = 1024, num_layers = 1)
        #self.balance_score = nn.Linear(2048,2)

        self.w_proj_layer_norm = LayerNorm(1024)

        self.w_proj = nn.Linear(1024, 1024)
        #self.w_context_vector = torch.randn([1024, 1]).float().requires_grad_() 
        self.w_context_vector = nn.Linear(1024, 1, bias = False)
        #nn.Parameter(torch.randn([1024, 1]).float(), requires_grad = True)
        
        #torch.rand(10).doube().requires_grad_() 
        
        self.softmax = nn.Softmax(dim = 1)
          
        #self.balance_score = nn.Sequential(
        #        nn.Linear(2048,2048),
        #       nn.Tanh(),
        #        nn.Linear(2048, 2)
        #)

        #self.softmax = nn.Softmax(dim = 1)

    @staticmethod
    def add_args(parser):
        super(BARTModel, BARTModel).add_args(parser)
        parser.add_argument(
            '--pooler-dropout', type=float, metavar='D',
            help='dropout probability in the masked_lm pooler layers'
        )
        parser.add_argument(
            '--pooler-activation-fn',
            choices=utils.get_available_activation_fns(),
            help='activation function to use for pooler layer'
        )

    @property
    def supported_targets(self):
        return {'self'}

    
    def forward_section_embeddings(self, embed, src_tokens):
        
        
        x = embed.transpose(1,0) + self.section_positions(src_tokens)
        
        x = self.section_layernorm_embedding(x)

        return x.transpose(1,0)

    def section_extract(self, src_tokens, encoder_out):
        
        sections = []

        #sections.append(self.encoder.embed_tokens(torch.tensor(self.encoder.dictionary.bos()).cuda()))
        
        encoder_out = encoder_out.transpose(0, 1)
        
        #max_length = torch.max(src_tokens.eq(1721).sum(dim = 1))
        
        #U = torch.rand(logits.shape)
        #if logits.is_cuda:
        #    U = U.cuda()
        #print(encoder_out)

        for i in range(0, src_tokens.shape[0]):
            #print(src_tokens[i])
            #segs = src_tokens[i].eq(24303) + src_tokens[i].eq(self.encoder.dictionary.eos())
            # 1721
            segs = src_tokens[i].eq(1721) + src_tokens[i].eq(15483)
            #src_tokens[i].eq(15483) + src_tokens[i].eq(1721) 
            if segs.sum() > 0:
                #print(segs)
                sections.append(encoder_out[i][segs])
            else:
                print(src_tokens[i])
                print('no segmentstions')
        
        
        
        #print(sections)
        sections = pad_sequence(sections)
        #print(sections)
        
        section_padding_mask = sections.eq(0).sum(-1) > 0
        section_padding = 1 - section_padding_mask.type(torch.long)
        
        #print(sections.shape)
        #print(sections[-1])

        return sections, section_padding_mask.transpose(1, 0), section_padding.transpose(1, 0)

    
    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, src2_tokens = None, src2_lengths = None,
        features_only=False, balance = False, classification_head_name=None, **kwargs
    ):
        if classification_head_name is not None:
            features_only = True


        #print("????", src_tokens.shape)

        #print("????", src_tokens)
        #print("!!!", src_lengths[0])
        #print("----", src2_tokens[0], src2_tokens[0].shape, src2_lengths[0])

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )

        sections, section_padding_mask, section_padding = self.section_extract(src_tokens, encoder_out.encoder_out)
        # T * B * C
        # B * T
        
        #print("???", sections)
        #section_out = sections
        #sections = self.forward_section_embeddings(sections, section_padding)
        #section_out = self.section(sections, src_key_padding_mask = section_padding_mask)
        section_out, _ = self.section(sections)
        #print("....", section_out.shape)
        #print(src2_tokens[0])

        #src2_tokens = None
        #balance = False
        
        if src2_tokens is not None:
            encoder_out2 = self.encoder(
                src2_tokens,
                src_lengths=src2_lengths,
                **kwargs,
            )
            
            
            sections2, section_padding_mask2, section_padding2 = self.section_extract(src2_tokens, encoder_out2.encoder_out)
            
            #sections2 = self.forward_section_embeddings(sections2, section_padding2)
            #section_out2 = sections2
            #section_out2 = self.section(sections2, src_key_padding_mask = section_padding_mask2)
            section_out2, _ = self.section(sections2)

        else:
            encoder_out2 = None
            sections2 = None
            section_out2 = None
            section_padding_mask2 = None
        
        #print(".....", encoder_out.encoder_out.shape)
        
        #print(".....", encoder_out.encoder_out.shape)

        if balance:

            #eos_mask1 = src_tokens.eq(2)
            #eos_mask2 = src2_tokens.eq(2)

            #print(src_tokens.shape)

            #print(encoder_out.encoder_out.shape)



            #src = encoder_out.encoder_out[src_tokens.eq(self.encoder.dictionary.eos()), :].view(encoder_out.encoder_out.size(0), -1, encoder_out.encoder_out.size(-1))[:, -1, :]

            #print(src.shape)

            #src2 = encoder_out2.encoder_out[src2_tokens.eq(self.encoder.dictionary.eos()), :].view(encoder_out2.encoder_out.size(0), -1, encoder_out2.encoder_out.size(-1))[:, -1, :]
            
            #print(src2.shape)

            #src = encoder_out.encoder_out[-1, :].unsqueeze(1)
            #src2 = encoder_out2.encoder_out[-1, :].unsqueeze(1)
            #src = torch.mean(section_out.transpose(1, 0), dim = 1).unsqueeze(1)
            #src2 = torch.mean(section_out2.transpose(1, 0), dim = 1).unsqueeze(1)

            #src = torch.nn.functional.max_pool1d(section_out.transpose(1, 0).transpose(2,1), kernel_size = section_out.shape[1]).unsqueeze(-1)
            #src2 = torch.nn.functional.max_pool1d(section_out2.transpose(1, 0).transpose(2,1), kernel_size = section_out2.shape[1]).unsqueeze(-1)
            src = section_out[-1, :].unsqueeze(1)
            src2 = section_out2[-1, :].unsqueeze(1)

            #print(src)
            #print(src2)
            #print("1")
            #print(sections)
            #print(src)

            #print('2')
            #print(sections2)
            #print(src2)
            
            #print(src.shape)

            #print(encoder_out.encoder_out[-1, :].shape)
            #print(src.shape)
            #print(src2.shape)
            
            src_input = torch.cat([src, src2], dim = 1)

            #print(src_input.shape)

            Hw = torch.tanh(self.w_proj_layer_norm(self.w_proj(src_input)))

            #print(Hw)
            #Hw = torch.tanh(self.w_proj(src_input))

            #balance_weight = self.softmax(Hw.matmul(self.w_context_vector).squeeze(-1))
            balance_weight = self.softmax(self.w_context_vector(Hw).squeeze(-1))

            #print(balance_weight)
            #print(self.args.T)
            #pt = p**(1/args.T)
            
            #self.T = self.args.T

            original_balance_weight = balance_weight



            balance_weight = balance_weight **(1/self.T)
            balance_weight = balance_weight/balance_weight.sum(dim = 1, keepdim = True)
            #targets_u = pt / pt.sum(dim=1, keepdim=True)

            #print(balance_weight.shape)
            #print(balance_weight)

            

            #balance_weight = self.softmax(self.balance_score(src_input))
            #print(balance_weight.shape)
            #print(balance_weight)
        else:
            balance_weight = None
            original_balance_weight = None



        #print('prev_output_token', prev_output_tokens)
        #print('encoder_out.encoder_out', encoder_out.encoder_out)

        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out= encoder_out,
            encoder_out2 = encoder_out2,
            features_only=features_only,
            balance_weight = balance_weight,
            **kwargs,
        )

        #print(".........")
        #print(x.shape)
        #print(x)

        '''
        x2, extra2 = self.decoder(
            prev_output_tokens,
            encoder_out= encoder_out2,
            encoder_out2 = encoder_out2,
            features_only=features_only,
            balance_weight = balance_weight,
            **kwargs,
        )
        '''

        #print(x)
        #print(x2)
        #x = 0.5 * x + 0.5 *  x2
        #x = x2

        #print("?????", x.shape)
        extra['original_balance_weight'] = original_balance_weight

        

        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(self.encoder.dictionary.eos()), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )
        return x, extra

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file='model.pt',
        data_name_or_path='.',
        bpe='gpt2',
        **kwargs,
    ):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        #print(x['task'], x['models'][0])
        return BARTHubInterface(x['args'], x['task'], x['models'][0])

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict['encoder.embed_tokens.weight'].size(0)
        if loaded_dict_size == len(self.encoder.dictionary) + 1 and '<mask>' not in self.encoder.dictionary:
            state_dict['encoder.embed_tokens.weight'] = state_dict['encoder.embed_tokens.weight'][:loaded_dict_size-1, :]
            state_dict['decoder.embed_tokens.weight'] = state_dict['decoder.embed_tokens.weight'][:loaded_dict_size-1, :]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture('bart', 'bart_large')
def bart_large_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4*1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', True)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
