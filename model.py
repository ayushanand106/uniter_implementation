import torch
import torch.nn as nn
import math
from apex.normalization.fused_layer_norm import FusedLayerNorm

# class LayerNormalization(nn.Module):

#     def __init__(self, features: int, eps:float=10**-6) -> None:
#         super().__init__()
#         self.eps = eps
#         self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
#         self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

#     def forward(self, x):
#         # x: (batch, seq_len, hidden_size)
#          # Keep the dimension for broadcasting
#         mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
#         # Keep the dimension for broadcasting
#         std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
#         # eps is to prevent dividing by zero or when std is very small
#         return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
# class ImageEmbeddings(nn.Module):
#     def __init__(self, config, img_dim):
#         super().__init__()
#         self.img_linear = nn.Linear(img_dim, config.hidden_size)
#         self.img_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
#         self.pos_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
#         self.pos_linear = nn.Linear(7, config.hidden_size)
#         self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

#         # tf naming convention for layer norm
#         self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, img_feat, img_pos_feat, type_embeddings, img_masks=None):
#         if img_masks is not None:
#             self.mask_embedding.weight.data[0, :].fill_(0)
#             mask = self.mask_embedding(img_masks.long())
#             img_feat = img_feat + mask

#         transformed_im = self.img_layer_norm(self.img_linear(img_feat))
#         transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
#         embeddings = transformed_im + transformed_pos + type_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings

# class UniterTextEmbeddings(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.word_embeddings = nn.Embedding(config.vocab_size,
#                                             config.hidden_size, padding_idx=0)
#         self.position_embeddings = nn.Embedding(config.max_position_embeddings,
#                                                 config.hidden_size)
#         self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
#                                                   config.hidden_size)

#         # self.LayerNorm is not snake-cased to stick with TensorFlow model
#         # variable name and be able to load any TensorFlow checkpoint file
#         self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, input_ids, position_ids, token_type_ids=None):
#         if token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids)

#         words_embeddings = self.word_embeddings(input_ids)
#         position_embeddings = self.position_embeddings(position_ids)
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)

#         embeddings = (words_embeddings
#                       + position_embeddings
#                       + token_type_embeddings)
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings



# class FeedForwardBlock(nn.Module):

#     def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
#         super().__init__()
#         self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
#         self.dropout = nn.Dropout(dropout)
#         self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

#     def forward(self, x):
#         # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
#         return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# # class InputEmbeddings(nn.Module):

# #     def __init__(self, d_model: int, vocab_size: int) -> None:
# #         super().__init__()
# #         self.d_model = d_model
# #         self.vocab_size = vocab_size
# #         self.embedding = nn.Embedding(vocab_size, d_model)

# #     def forward(self, x):
# #         # (batch, seq_len) --> (batch, seq_len, d_model)
# #         # Multiply by sqrt(d_model) to scale the embeddings according to the paper
# #         return self.embedding(x) * math.sqrt(self.d_model)
    
# # class PositionalEncoding(nn.Module):

# #     def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
# #         super().__init__()
# #         self.d_model = d_model
# #         self.seq_len = seq_len
# #         self.dropout = nn.Dropout(dropout)
# #         # Create a matrix of shape (seq_len, d_model)
# #         pe = torch.zeros(seq_len, d_model)
# #         # Create a vector of shape (seq_len)
# #         position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
# #         # Create a vector of shape (d_model)
# #         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
# #         # Apply sine to even indices
# #         pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
# #         # Apply cosine to odd indices
# #         pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
# #         # Add a batch dimension to the positional encoding
# #         pe = pe.unsqueeze(0) # (1, seq_len, d_model)
# #         # Register the positional encoding as a buffer
# #         self.register_buffer('pe', pe)

# #     def forward(self, x):
# #         x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
# #         return self.dropout(x)

# class ResidualConnection(nn.Module):
    
#         def __init__(self, features: int, dropout: float) -> None:
#             super().__init__()
#             self.dropout = nn.Dropout(dropout)
#             self.norm = LayerNormalization(features)
    
#         def forward(self, x, sublayer):
#             return x + self.dropout(sublayer(self.norm(x)))

# class MultiHeadAttentionBlock(nn.Module):

#     def __init__(self, d_model: int, h: int, dropout: float) -> None:
#         super().__init__()
#         self.d_model = d_model # Embedding vector size
#         self.h = h # Number of heads
#         # Make sure d_model is divisible by h
#         assert d_model % h == 0, "d_model is not divisible by h"

#         self.d_k = d_model // h # Dimension of vector seen by each head
#         self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
#         self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
#         self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
#         self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
#         self.dropout = nn.Dropout(dropout)

#     @staticmethod
#     def attention(query, key, value, mask, dropout: nn.Dropout):
#         d_k = query.shape[-1]
#         # Just apply the formula from the paper
#         # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
#         attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
#         if mask is not None:
#             # Write a very low value (indicating -inf) to the positions where mask == 0
#             attention_scores.masked_fill_(mask == 0, -1e9)
#         attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
#         if dropout is not None:
#             attention_scores = dropout(attention_scores)
#         # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
#         # return attention scores which can be used for visualization
#         return (attention_scores @ value), attention_scores

#     def forward(self, q, k, v, mask):
#         query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
#         key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
#         value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

#         # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
#         query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
#         key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
#         value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

#         # Calculate attention
#         x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
#         # Combine all the heads together
#         # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
#         x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

#         # Multiply by Wo
#         # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
#         return self.w_o(x)

# class EncoderBlock(nn.Module):

#     def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
#         super().__init__()
#         self.self_attention_block = self_attention_block
#         self.feed_forward_block = feed_forward_block
#         self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

#     def forward(self, x, src_mask):
#         x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
#         x = self.residual_connections[1](x, self.feed_forward_block)
#         return x
    
# class Encoder(nn.Module):

#     def __init__(self, features: int, layers: nn.ModuleList) -> None:
#         super().__init__()
#         self.layers = layers
#         self.norm = LayerNormalization(features)

#     def forward(self, x, mask):
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)

# class DecoderBlock(nn.Module):

#     def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
#         super().__init__()
#         self.self_attention_block = self_attention_block
#         self.cross_attention_block = cross_attention_block
#         self.feed_forward_block = feed_forward_block
#         self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

#     def forward(self, x, encoder_output, src_mask, tgt_mask):
#         x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
#         x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
#         x = self.residual_connections[2](x, self.feed_forward_block)
#         return x
    
# class Decoder(nn.Module):

#     def __init__(self, features: int, layers: nn.ModuleList) -> None:
#         super().__init__()
#         self.layers = layers
#         self.norm = LayerNormalization(features)

#     def forward(self, x, encoder_output, src_mask, tgt_mask):
#         for layer in self.layers:
#             x = layer(x, encoder_output, src_mask, tgt_mask)
#         return self.norm(x)

# class ProjectionLayer(nn.Module):

#     def __init__(self, d_model, vocab_size) -> None:
#         super().__init__()
#         self.proj = nn.Linear(d_model, vocab_size)

#     def forward(self, x) -> None:
#         # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
#         return self.proj(x)
    
# class Transformer(nn.Module):

#     def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.src_pos = src_pos
#         self.tgt_pos = tgt_pos
#         self.projection_layer = projection_layer

#     def encode(self, src, src_mask):
#         # (batch, seq_len, d_model)
#         src = self.src_embed(src)
#         src = self.src_pos(src)
#         return self.encoder(src, src_mask)
    
#     def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
#         # (batch, seq_len, d_model)
#         tgt = self.tgt_embed(tgt)
#         tgt = self.tgt_pos(tgt)
#         return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
#     def project(self, x):
#         # (batch, seq_len, vocab_size)
#         return self.projection_layer(x)
    
# def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
#     # Create the embedding layers
#     src_embed = InputEmbeddings(d_model, src_vocab_size)
#     tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

#     # Create the positional encoding layers
#     src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
#     tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
#     # Create the encoder blocks
#     encoder_blocks = []
#     for _ in range(N):
#         encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
#         feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
#         encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
#         encoder_blocks.append(encoder_block)

#     # Create the decoder blocks
#     decoder_blocks = []
#     for _ in range(N):
#         decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
#         decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
#         feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
#         decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
#         decoder_blocks.append(decoder_block)
    
#     # Create the encoder and decoder
#     encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
#     decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
#     # Create the projection layer
#     projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
#     # Create the transformer
#     transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
#     # Initialize the parameters
#     for p in transformer.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
    
#     return transformer


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(
            torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class UniterTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings
                      + position_embeddings
                      + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.pos_linear = nn.Linear(7, config.hidden_size)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, type_embeddings, img_masks=None):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, input_, attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        hidden_states = input_
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class UniterModel(UniterPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.embeddings = UniterTextEmbeddings(config)
        self.img_embeddings = UniterImageEmbeddings(config, img_dim)
        self.encoder = UniterEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, img_masks=None,
                                img_type_ids=None):
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.embeddings.token_type_embeddings(
            img_type_ids)
        output = self.img_embeddings(img_feat, img_pos_feat,
                                     img_type_embeddings, img_masks)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    gather_index, img_masks=None,
                                    txt_type_ids=None, img_type_ids=None):
        txt_emb = self._compute_txt_embeddings(
            input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_pos_feat, img_masks, img_type_ids)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def forward(self, input_ids, position_ids,
                img_feat, img_pos_feat,
                attention_mask, gather_index=None, img_masks=None,
                output_all_encoded_layers=True,
                txt_type_ids=None, img_type_ids=None):
        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids)
        else:
            embedding_output = self._compute_img_txt_embeddings(
                input_ids, position_ids,
                img_feat, img_pos_feat,
                gather_index, img_masks, txt_type_ids, img_type_ids)

        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = UniterTextEmbeddings(d_model, src_vocab_size)
    tgt_embed = UniterImageEmbeddings(d_model, tgt_vocab_size)

    # # Create the positional encoding layers
    # src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    # tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer