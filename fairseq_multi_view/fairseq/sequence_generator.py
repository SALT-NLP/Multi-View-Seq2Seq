# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder

import pickle


class SequenceGenerator(object):
    def __init__(
        self,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        temperature=1.,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert temperature > 0, '--temperature must be greater than 0'

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )


    @torch.no_grad()
    def generate(self, models, sample, balance = False, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        model = EnsembleModel(models)
        return self._generate(model, sample, balance = balance, **kwargs)

    @torch.no_grad()
    def _generate(
        self,
        model,
        sample,
        prefix_tokens=None,
        bos_token=None,
        balance = False,
        **kwargs
    ):
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()

        if 'src2_tokens' in encoder_input:
            src_tokens2 = encoder_input['src2_tokens']
            src_lengths2 = (src_tokens2.ne(self.eos) & src_tokens2.ne(self.pad)).long().sum(dim=1)
            input_size2 = src_tokens2.size()
        else:
            src_tokens2 = None
            src_lengths2 = None
            input_size2 = None

        
        
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        if input_size2 is not None:
            src_len = max(input_size[1], input_size2[1])
        else:
            src_len = input_size[1]

        if 'src2_tokens' in encoder_input:
            src_len2 = input_size2[1]

        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )
        
        

        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!'

        # compute the encoder output for each beam
        encoder_out = model.forward_encoder(encoder_input, balance)

        if encoder_out[1] == None:
            encoder_outs = encoder_out[0]

            new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
            
            new_order = new_order.to(src_tokens.device).long()
            
            encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)
            
            encoder_outs2 = None
            balance_weight = None

        else:
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
            new_order = new_order.to(src_tokens.device).long()
            
            #print(new_order)
            #print("....", encoder_out[0][0].encoder_out.shape)
            encoder_outs = model.reorder_encoder_out(encoder_out[0], new_order)
            #print("finished")
            #print("????", encoder_out[1][0].encoder_out.shape)
            encoder_outs2 = model.reorder_encoder_out(encoder_out[1], new_order)
            
            balance_weight = encoder_out[2]
            if encoder_out[2] is None:
                balance_weight = None
            else:
                #print(balance_weight)
                balance_weight = balance_weight.index_select(0, new_order)
                #print(balance_weight)
        # TODO: Modify this
        # initialize buffers

        #if encoder_out[1] is not None and encoder_outs[2] is None:
        #    src_tokens = torch.cat([src_tokens, src_tokens2], dim = 1)
        

        scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        
        tokens = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        
        #if encoder_out[2] is not None:
        #    scores2 = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        #    scores_buf2 = scores2.clone()
            
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        
        attn, attn_buf = None, None

        attn2, attn_buf2 = None, None

        # The blacklist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the blacklist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        blacklist = src_tokens.new_zeros(bsz, beam_size).eq(-1)  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size or step == max_len:
                return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            assert not tokens_clone.eq(self.eos).any()
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i]
                    else:
                        hypo_attn = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': None,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                
                #TODO:
                model.reorder_incremental_state(reorder_state)
                
                #TODO:   
                 
                encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)
                  
                if encoder_outs2 is not None:
                    encoder_outs2 = model.reorder_encoder_out(encoder_outs2, reorder_state)

                if balance_weight is not None:
                    balance_weight = balance_weight.index_select(0, reorder_state)

            if balance_weight is not None: 
                #print(lprobs)
                #print("out 0", encoder_outs[0].encoder_out)
                #print("out 1", encoder_outs2[0].encoder_out)
                #print(step)
                
                lprobs, avg_attn_scores, avg_attn_scores2, balance_weight = model.forward_decoder(
                tokens[:, :step + 1], encoder_outs, temperature=self.temperature, encoder_outs2 = encoder_outs2, balance_weight = balance_weight,
            )
                
                #break

            elif encoder_outs2 is not None:
                lprobs, avg_attn_scores = model.forward_decoder(
                    tokens[:, :step + 1], encoder_outs, temperature=self.temperature, encoder_outs2 = encoder_outs2, balance_weight = balance_weight,
                )
                avg_attn_scores2 = None
            else:
                lprobs, avg_attn_scores = model.forward_decoder(
                    tokens[:, :step + 1], encoder_outs, temperature=self.temperature,
                )
                avg_attn_scores2 = None

            lprobs[lprobs != lprobs] = -math.inf

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
                prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
                prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.pad)
                lprobs[prefix_mask] = -math.inf
                lprobs[prefix_mask] = lprobs[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
                )
                # if prefix includes eos, then we should make sure tokens and
                # scores are the same across all beams
                eos_mask = prefix_toks.eq(self.eos)
                if eos_mask.any():
                    # validate that the first beam matches the prefix
                    first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
                    eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
                    target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
                    assert (first_beam == target_prefix).all()

                    def replicate_first_beam(tensor, mask):
                        tensor = tensor.view(-1, beam_size, tensor.size(-1))
                        tensor[mask] = tensor[mask][:, :1, :]
                        return tensor.view(-1, tensor.size(-1))

                    # copy tokens, scores and lprobs from the first beam to all beams
                    tokens = replicate_first_beam(tokens, eos_mask_batch_dim)
                    scores = replicate_first_beam(scores, eos_mask_batch_dim)
                    lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim)
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if type(avg_attn_scores) is list:
                avg_attn_scores = avg_attn_scores[0]
            
            if type(avg_attn_scores2) is list:
                avg_attn_scores2 = avg_attn_scores2[0]

            
            if avg_attn_scores is not None:
                if encoder_outs2 is None or balance_weight is not None:
                    if attn is None:
                        attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                        attn_buf = attn.clone()
                    attn[:, :, step + 1].copy_(avg_attn_scores)
                else:
                    if attn is None:
                        attn = scores.new(bsz * beam_size, src_tokens.size(1) + src_tokens2.size(1), max_len + 2)
                        attn_buf = attn.clone()
                    attn[:, :, step + 1].copy_(avg_attn_scores)

            if avg_attn_scores2 is not None:
                if attn2 is None:
                    attn2 = scores.new(bsz * beam_size, src_tokens2.size(1), max_len + 2)
                    attn_buf2 = attn2.clone()
                attn2[:, :, step + 1].copy_(avg_attn_scores2)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)


            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)

            self.search.set_src_lengths(src_lengths)


            if self.no_repeat_ngram_size > 0:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                    return gen_ngrams[bbsz_idx].get(ngram_index, [])

                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
                else:
                    banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                for bbsz_idx in range(bsz * beam_size):
                    lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf

            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos, except for blacklisted ones
            # or candidates with a score of -inf
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][blacklist] = 0

            # only consider eos when it's among the top beam_size indices
            torch.masked_select(
                cand_bbsz_idx[:, :beam_size],
                mask=eos_mask[:, :beam_size],
                out=eos_bbsz_idx,
            )

            finalized_sents = set()
            if eos_bbsz_idx.numel() > 0:
                torch.masked_select(
                    cand_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_scores,
                )
                finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores)
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                
                if attn2 is not None:
                    attn2 = attn2.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn2.size(1), -1)
                    attn_buf2.resize_as_(attn2)

                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos or
            # blacklisted hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.
            active_mask = buffer('active_mask')
            eos_mask[:, :beam_size] |= blacklist
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(new_blacklist, active_hypos)
            )

            # update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(cand_size)[:, :beam_size]
            assert (~blacklist).any(dim=1).all()

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn
            
            if attn2 is not None:
                attn2, attn_buf2 = attn_buf2, attn2

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)
        return finalized


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(hasattr(m, 'decoder') and isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}


        self.incremental_states2 = None
        if all(hasattr(m, 'decoder') and isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states2 = {m: {} for m in models}

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input, balance = False):
        if not self.has_encoder():
            return None
        if 'src2_tokens' not in encoder_input:
            
            return [model.encoder(**encoder_input) for model in self.models], None, None
        else:

            src_tokens = encoder_input['src_tokens']
            src_lengths = encoder_input['src_lengths']
            src2_tokens = encoder_input['src2_tokens']
            src2_length = encoder_input['src2_lengths']

            #print(src_lengths, src2_length)

            encoder_out = self.models[0].encoder(src_tokens, src_lengths=src_lengths)
            encoder_out2 = self.models[0].encoder(src2_tokens, src_lengths=src2_length)

            sections, section_padding_mask, section_padding = self.models[0].section_extract(src_tokens, encoder_out.encoder_out)
            
            #sections = self.models[0].forward_section_embeddings(sections, section_padding)
            #section_out = self.models[0].section(sections, src_key_padding_mask = section_padding_mask)
            #section_out = sections
            section_out, _ = self.models[0].section(sections)
            
            
            sections2, section_padding_mask2, section_padding2 = self.models[0].section_extract(src2_tokens, encoder_out2.encoder_out)
            
            #sections2 = self.models[0].forward_section_embeddings(sections2, section_padding2)
            #section_out2 = self.models[0].section(sections2, src_key_padding_mask = section_padding_mask2)
            #section_out2 = sections2
            section_out2, _ = self.models[0].section(sections2)

            #print(balance)

            if balance:
                #eos_mask1 = src_tokens.eq(2)
                #eos_mask2 = src2_tokens.eq(2)

                #print(src_tokens)
                #print(src2_tokens)

                #src = encoder_out.encoder_out[-1, :].unsqueeze(1)
                #src2 = encoder_out2.encoder_out[-1, :].unsqueeze(1)
                
                #src = torch.mean(section_out.transpose(1, 0), dim = 1).unsqueeze(1)
                #src2 = torch.mean(section_out2.transpose(1, 0), dim = 1).unsqueeze(1)
                src = section_out[-1, :].unsqueeze(1)
                src2 = section_out2[-1, :].unsqueeze(1)
                #src = torch.nn.functional.max_pool1d(section_out.transpose(1, 0).transpose(2,1), kernel_size = section_out.shape[1]).unsqueeze(-1)
                #src2 = torch.nn.functional.max_pool1d(section_out2.transpose(1, 0).transpose(2,1), kernel_size = section_out2.shape[1]).unsqueeze(-1)

                src_input = torch.cat([src, src2], dim = 1)
                Hw = torch.tanh(self.models[0].w_proj_layer_norm(self.models[0].w_proj(src_input)))
                #Hw = torch.tanh(self.models[0].w_proj(src_input))

                #balance_weight = self.models[0].softmax(Hw.matmul(self.models[0].w_context_vector).squeeze(-1))
                balance_weight = self.models[0].softmax(self.models[0].w_context_vector(Hw).squeeze(-1))

                #print('before sharpen', balance_weight)

                balance_weight = balance_weight **(1/self.models[0].args.T)
                balance_weight = balance_weight/balance_weight.sum(dim = 1, keepdim = True)

                #print("......")
            else:
                balance_weight = None

            
            #print(self.models[0].w_proj.weight)
            #print(self.models[0].w_proj.bias)
            #print("------")
            #print("before tanh", self.models[0].w_proj(src_input))
            #print("after layer norm", self.models[0].w_proj_layer_norm(self.models[0].w_proj(src_input)))
            #print(Hw)
            #print("-----")
            #print(self.models[0].w_context_vector(Hw))

            #print('0')
            #print(sections)
            #print(section_out)
            #print(src)
            
           
            #print('1')
            #print(sections2)
            #print(section_out2)
            #print(src2)
           
            #print(balance_weight)

            return ([encoder_out], [encoder_out2], balance_weight)

            


    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1., encoder_outs2 = None, balance_weight = None):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
                encoder_outs2 = encoder_outs2[0] if encoder_outs2 is not None else None,
                balance_weight = balance_weight,
                incremental_states2 = self.incremental_states2
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1., encoder_outs2 = None, balance_weight = None, incremental_states2 = None
    ):
        if self.incremental_states is not None:
            #print("not None")
            decoder_out = list(model.forward_decoder(
                tokens, encoder_out=encoder_out, incremental_state=self.incremental_states[model], encoder_out2 = encoder_outs2, balance_weight = balance_weight, incremental_state2 = incremental_states2,
            ))

        else:
            print('None')
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out, encoder_out2 = encoder_outs2, balance_weight = balance_weight,))

           
        
        #print(decoder_out[0].shape)
        #print(decoder_out2[0].shape)

        decoder_out[0] = decoder_out[0][:, -1:, :]

        #print(decoder_out[0])

        #print(decoder_out[0].shape)

        if temperature != 1.:
            decoder_out[0].div_(temperature)

        
        attn = decoder_out[1]

        #print(attn['attn2'])
        #print(attn['balance_weight'])

        if type(attn) is dict:
            attn2 = attn.get('attn2', None)
            balance_weight = attn.get('balance_weight', None)
            attn = attn.get('attn', None)


        if type(attn) is list:
            attn = attn[0]

        if type(attn2) is list:
            attn2 = attn2[0]


        if attn is not None:
            #print(attn.shape)
            attn = attn[:, -1, :]
            #print(attn.shape)
        
        if attn2 is not None:
            #print(attn2.shape)
            attn2 = attn2[:, -1, :]

        if balance_weight is not None:
            #print(balance_weight.shape)
            balance_weight = balance_weight
        
        #print(balance_weight[-1])
        
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        if attn2 is not None:
            return probs, attn, attn2, balance_weight
        else:
            return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            print('None')
            return
        
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)

        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states2[model], new_order)


class SequenceGeneratorWithAlignment(SequenceGenerator):

    def __init__(self, tgt_dict, left_pad_target=False, **kwargs):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        model = EnsembleModelWithAlignment(models)
        finalized = super()._generate(model, sample, **kwargs)

        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        src_tokens, src_lengths, prev_output_tokens, tgt_tokens = \
            self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, 'full_context_alignment', False) for m in model.models):
            attn = model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [
                finalized[i // beam_size][i % beam_size]['attention'].transpose(1, 0)
                for i in range(bsz * beam_size)
            ]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            alignment = utils.extract_hard_alignment(attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos)
            finalized[i // beam_size][i % beam_size]['alignment'] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        src_tokens = src_tokens[:, None, :].expand(-1, self.beam_size, -1).contiguous().view(bsz * self.beam_size, -1)
        src_lengths = sample['net_input']['src_lengths']
        src_lengths = src_lengths[:, None].expand(-1, self.beam_size).contiguous().view(bsz * self.beam_size)
        prev_output_tokens = data_utils.collate_tokens(
            [beam['tokens'] for example in hypothesis for beam in example],
            self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam['tokens'] for example in hypothesis for beam in example],
            self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]['attn']
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens,
                encoder_out=encoder_out,
                incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if type(attn) is list:
            attn = attn[0]
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn
