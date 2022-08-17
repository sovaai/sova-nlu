import torch
from torch import nn as nn
from torch import Tensor
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional
from typing import Optional, List
from collections import OrderedDict


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(
        self,
        vocab_size: int,
        token_type_vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        hidden_dropout_prob: int,
        pad_token_id: int,
        layer_norm_eps: float = 1e-12,
        position_embedding_type = 'absolute'
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(token_type_vocab_size, hidden_size)

        embedding_qconfig = torch.quantization.float_qparams_weight_only_qconfig
        self.word_embeddings.qconfig = embedding_qconfig
        self.position_embeddings.qconfig = embedding_qconfig
        self.token_type_embeddings.qconfig = embedding_qconfig

        self.embeddings_quant = QuantStub()
        self.embeddings_dequant = DeQuantStub()

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layer_norm.qconfig = None
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.absolute_embedding_type: bool = position_embedding_type == 'absolute'
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        # registering buffer is required to be able to not pass `token_type_ids` on forward (as in most BERT usecases)
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, :seq_length].clone().contiguous()

        if token_type_ids is None:
            buffered_token_type_ids = self.token_type_ids[:, :seq_length].clone()
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        token_type_embeddings = self.token_type_embeddings(token_type_ids.contiguous())

        inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.absolute_embedding_type:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.embeddings_quant(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_probs_dropout_prob: float,
                 position_embedding_type: str,
                 max_position_embeddings: int):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.register_buffer(
            'attention_normalizer',
            torch.sqrt(
                torch.tensor(
                    self.attention_head_size,
                )
            ),
            persistent=False
        )

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.query_dequant = DeQuantStub()
        self.key_dequant = DeQuantStub()
        self.value_dequant = DeQuantStub()

        self.context_quant = QuantStub()

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * self.max_position_embeddings - 1, self.attention_head_size)
            if self.position_embedding_type == "relative_key":
                self.forward = self.forward_relative_key_position_embedding
            else:
                self.forward = self.forward_relative_key_query_position_embedding
        else:
            self.forward = self.forward_absolute_position_embedding

    def transpose_for_scores(self, x: Tensor):
        new_x_shape = x.size()
        x = x.view(new_x_shape[0], new_x_shape[1], self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward_absolute_position_embedding(
        self,
        hidden_states,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        key_layer = self.transpose_for_scores(self.key_dequant(self.key(hidden_states)))
        value_layer = self.transpose_for_scores(self.value_dequant(self.value(hidden_states)))
        query_layer = self.transpose_for_scores(self.query_dequant(self.query(hidden_states)))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / self.attention_normalizer.clone().contiguous()
        # attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size))

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.softmax(attention_scores, dim=-1)


        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.context_quant(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape[0], new_context_layer_shape[1], new_context_layer_shape[2])

        outputs = [context_layer, attention_probs] if output_attentions is not None and output_attentions else [context_layer]

        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size: int, hidden_dropout_prob: float, layer_norm_eps: float):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.residual_add_float_func = FloatFunctional()

    def forward(self, hidden_states: Tensor, input_tensor: Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.residual_add_float_func.add(hidden_states, input_tensor)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_dropout_prob: float,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        position_embedding_type: str,
        max_position_embeddings: int,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, position_embedding_type, max_position_embeddings)
        self.output = BertSelfOutput(hidden_size, hidden_dropout_prob, layer_norm_eps)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = [attention_output]
        outputs.extend(self_outputs[1:])  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            if hidden_act == 'gelu':
                self.intermediate_act_fn = torch.nn.GELU()
            else:
                raise ValueError(f'Unsupported activation function `{hidden_act}`')
        else:
            self.intermediate_act_fn = hidden_act

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, hidden_states: Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dequant(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.quant(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.residual_add_float_func = FloatFunctional()

    def forward(self, hidden_states: Tensor, input_tensor: Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.residual_add_float_func.add(hidden_states, input_tensor)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        hidden_act,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        layer_norm_eps,
        position_embedding_type,
        max_position_embeddings,
    ):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = BertAttention(
            hidden_size,
            hidden_dropout_prob,
            num_attention_heads,
            attention_probs_dropout_prob,
            position_embedding_type,
            max_position_embeddings,
            layer_norm_eps,
        )
        self.intermediate = BertIntermediate(hidden_size, intermediate_size, hidden_act)
        self.output = BertOutput(intermediate_size, hidden_size, hidden_dropout_prob, layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        # hidden_states = self.hidden_states_quant(hidden_states)
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_outputs = self.output(self.intermediate(attention_output), attention_output)
        # layer_outputs = self.hidden_states_dequant(layer_outputs)

        outputs = [layer_outputs]
        outputs.extend(self_attention_outputs[1:])
        return outputs


class BertEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_act: str,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        layer_norm_eps: float,
        gradient_checkpointing: bool,
        position_embedding_type: str,
        max_position_embeddings: int,
    ):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(hidden_size,
                                              num_attention_heads,
                                              intermediate_size,
                                              hidden_act,
                                              hidden_dropout_prob,
                                              attention_probs_dropout_prob,
                                              layer_norm_eps,
                                              position_embedding_type,
                                              max_position_embeddings) for _ in range(num_hidden_layers)])
        self.gradient_checkpointing = gradient_checkpointing

        @torch.jit.unused
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, past_key_value, output_attentions)

            return custom_forward
        self.create_custom_forward = create_custom_forward

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
    ):
        all_hidden_states = [] if output_hidden_states is not None and output_hidden_states else None
        all_self_attentions = [] if output_attentions is not None and output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states is not None and output_hidden_states and all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if torch.jit.is_scripting():
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                if self.gradient_checkpointing and self.training :
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        self.create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        output_attentions
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        output_attentions,
                    )

            hidden_states = layer_outputs[0]
            if output_attentions is not None and output_attentions and all_self_attentions is not None:
                all_self_attentions = all_self_attentions.append(layer_outputs[1])

        if output_hidden_states is not None and output_hidden_states and all_hidden_states is not None:
            all_hidden_states = all_hidden_states.append(hidden_states)

        return (
            hidden_states,
            all_hidden_states,
            all_self_attentions,
        )


class BertModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_type_vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        hidden_act: str,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        max_position_embeddings: int,
        pad_token_id: int,
        position_embedding_type: str = 'absolute',
        layer_norm_eps: float = 10e-12,
        gradient_checkpointing: bool = False
    ):
        super(BertModel, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.embeddings = BertEmbeddings(
            vocab_size,
            token_type_vocab_size,
            hidden_size,
            max_position_embeddings,
            hidden_dropout_prob,
            pad_token_id,
            layer_norm_eps,
            position_embedding_type)
        self.encoder = BertEncoder(
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            layer_norm_eps,
            gradient_checkpointing,
            position_embedding_type,
            max_position_embeddings,
        )

    def init_weights(self, initialization_range):
        """Initialize the weights"""
        def init(module):
            if isinstance(module, nn.Linear):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=initialization_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=initialization_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        self.apply(init)

    def load_huggingface_weights(self, foreign_model):
        state_dict = foreign_model
        dst_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if 'pooler' in key:
                continue

            new_key = key.replace('LayerNorm', 'layer_norm')
            dst_state_dict[new_key] = value
        self.load_state_dict(dst_state_dict, strict=False)

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: List[int], device: torch.device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.
        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_heads: int, num_hidden_layers: int, dtype: torch.dtype, device: torch.device, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.
        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
        if head_mask is None:
            head_mask = torch.ones((num_heads), dtype=torch.float, device=device)
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask: Tensor, num_hidden_layers: int):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=head_mask.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), dtype=torch.float, device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length].clone().contiguous()
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            head_mask = self.get_head_mask(head_mask, self.num_attention_heads, self.num_hidden_layers,
                                       attention_mask.dtype, device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]

        return sequence_output, encoder_outputs[1:]
