import torch
from torch import nn
from typing import Tuple, Optional, List
from torch.jit import ignore
from torch.nn import CrossEntropyLoss, attention
import math
from siglip_model import SiglipVisionConfig, SiglipVisionModel


class KVCache():
    
    def __init__(self):
        self.key_cache : List[torch.Tensor] = []
        self.value_cache : List[torch.Tensor] = []

    def num_items(self)-> int:
        if len(self.key_cache)==0:
            return 0
        else:
            # shape of the key_cache is [btach_size, num_heads_kv, seq_len, head_dim] i.e., return seq_len
            return self.key_cache[0].shape[-2]

    def update(self,key_states:torch.Tensor, value_states: torch.Tensor, layer_idx):
        
        if len(self.key_cache)<= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim= -2) #concat along seq dimension
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx],value_states],dim= -2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GemmaConfig():
    
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim = 256,
        max_position_embeddings = 8192,
        rms_norm_eps = 1e-6,
        rope_theta = 10000.0,
        attention_bias = False,
        attention_dropout = 0.0,
        pad_token_id = None,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id



class PaliGemmaConfig():
    def __init__(
        self,
        vision_config = None,
        text_config = None,
        ignore_index = -100,
        image_token_index = 256000,
        vocab_size = 257152,
        projection_dim = 2048,
        hidden_size = 2048,
        pad_token_id = None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id = pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size//self.vision_config.patch_size)**2
        self.vision_config.projection_dim = projection_dim


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config:PaliGemmaConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self,image_features):
        hidden_states = self.linear(image_features)
        return hidden_states

class GemmaRMSNorm(nn.Module):
    def __init__(self,dim: int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self,x):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim = True)+self.eps)
    
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1+self.weight.float())
        return output.type_as(x)

class GemmaMLP(nn.Module):

    def __init__(self, config:GemmaConfig):
        super().__init__()
        
        self.hidden_size = config.hidden_size

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size,bias=False) # to be used by activation function to make activation learnable
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size,config.hidden_size,bias=False)

    def forward(self, hidden_states):
        hidden_states = self.down_proj(nn.functional.gelu(self.gate_proj(hidden_states),approximate="tanh")*self.up_proj(hidden_states))
        return hidden_states

def repeat_kv(hidden_states: torch.Tensor, n_rep: int):
    batch, num_key_value_heads, seqlen, head_dim = hidden_states.shape

    if n_rep ==1:
        return hidden_states
    
    hidden_states = hidden_states[:,:,None,:,:].expand(batch, num_key_value_heads, n_rep, seqlen, head_dim)

    return hidden_states.reshape(batch, num_key_value_heads*n_rep,seqlen,head_dim)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings= 2048, base = 10000, device = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0/ (self.base ** (torch.arange(0,self.dim,2, dtype= torch.int64).float() / self.dim))
        
        self.register_buffer("inv_freq", tensor= inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len = None):
        #x -> [batch-size, numm_attention_heads, seqlen, head_size]
        self.inv_freq.to(x.device)
        #copy / repeat inv_freq for the batch in the seq
        # inv_freq_expands -> [btach_size, head_dim//2, 1]
        inv_freq_expanded = self.inv_freq[None,:,None].float().expand(position_ids.shape[0], -1 , 1)
        
        #expand position ids
        position_ids_expanded = position_ids[:,None,:].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type = device_type, enabled=False):
            #multiply each theta by position which is the argument of cos and sin
            #freqs: [batch-size, head_dim//2 , 1] @ [batch_size, 1, seq_len] -> [batch, seq, head_dim//2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,2)
            #emb -> [batch, seqlen, headdim]
            emb = torch.cat((freqs, freqs), dim= -1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(x.dtype), sin.to((x.dtype))

def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat((-x2, x1), dim= -1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1):
    cos = cos.unsqueeze(unsqueeze_dim) # adding head dimension
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q*cos + (rotate_half(q)*sin)
    k_embed = k*cos + (rotate_half(k)*sin)

    return q_embed, k_embed



class GemmaAttention(nn.Module):
    
    def __init__(self, config: GemmaConfig, layer_idx:Optional[int]=None):

        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.attention_dropout = config.attention_dropout
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads//self.num_key_value_heads
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_casual = True
        
        assert self.hidden_size % self.num_heads ==0

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias = config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias= config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias= config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings = self.max_position_embeddings,
            base = self.rope_theta
        )

    
    def forward(
        self,
        hidden_states:torch.Tensor,
        attention_mask:Optional[torch.FloatTensor]=None,
        position_ids:Optional[torch.LongTensor] = None,
        kv_cache:Optional[KVCache]=None,
        **kwargs,

    )->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batchsize, qlen, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
 
        #[batch-size, num_heads_q, seqlen, head_dim]
        query_states = query_states.view(batchsize,qlen, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batchsize,qlen, self.num_key_value_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batchsize,qlen, self.num_key_value_heads, self.head_dim).transpose(1,2)

        #[batchsize, seq_len, headDim] each
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len = None)
        #[batch_size, numhead_q, seq_len, head_dim], [batch_size, numHead_kv, seqlen, headdim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx) # to save the key and values states for this layer in the cache

        #to repeat kv for each query inside the group

        key_states = repeat_kv(key_states,self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None

        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim= -1, dtype= torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p = self.attention_dropout, training=self.training)

        #[batch_size, numhead_q, seq_len_q, seq_len_kv] x [btach_size, num_heads_kv, seq_len_kv, head_dim] = 
        attn_output = torch.matmul(attn_weights, value_states)

        if(attn_output.size() != (batchsize, self.num_heads, qlen, self.head_dim)):
            raise ValueError(
                f"'atten_out' should be of size {(batchsize, self.num_heads, qlen, self.head_dim)}, but is "
                f"{attn_output.size()}"
            )
            #to make sqlen the second dimension
        attn_output = attn_output.transpose(1,2).contiguous()
        #concat all the heads together [batch_size, qlen, num_heads_q, head_dim]-> [batchsize, qlen, num_heads_q * head_dim]
        attn_output = attn_output.view(batchsize, qlen, -1)
        # mixing results of different heads
        attn_output = self.o_proj(attn_output)

        return (attn_output, attn_weights)





class GemmaDecoderLayer(nn.Module):
    def __init__(self,config: GemmaConfig,layer_idx : int):
        super().__init__()
        self.self_attn = GemmaAttention(config =config, layer_idx = layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps= config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states:torch.Tensor,
        attention_mask:Optional[torch.Tensor]=None,
        position_ids:Optional[torch.LongTensor]=None,
        kv_cache:Optional[KVCache]=None,

    )->Tuple[torch.FloatTensor,Optional[Tuple[torch.FloatTensor,torch.FloatTensor]]]:

        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states,_, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            kv_cache = kv_cache
        )

        hidden_states = hidden_states+residual

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states+residual

        return hidden_states




class GemmaModel(nn.Module):

    def __init__(self,config:GemmaConfig):
        super().__init__()
        self.config = config
        self.pading_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size,self.pading_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config,layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask:Optional[torch.Tensor]=None,
        position_ids:Optional[torch.LongTensor]=None,
        inputs_embeds:Optional[torch.FloatTensor]=None,
        kv_cache:Optional[KVCache]=None,

    )->torch.FloatTensor:
        #[batch_size, seqlen, hidden_size] during all the operation shape didnt change
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype= hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                kv_cache = kv_cache,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states




class GemmaForCasualLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)


    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask:Optional[torch.Tensor] = None,
        position_ids:Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    )-> Tuple:
        outputs = self.model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            kv_cache = kv_cache,
        )
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = { "logits":logits}

        if kv_cache is not None:

            return_data['kv_cache'] = kv_cache

        return return_data


class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self,config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        
        self.language_model = GemmaForCasualLM(config.text_config)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    
    def tie_weights(self):
        return self.language_model.tie_weights()  # share some parameters between the layers of the transfoermer model
    
    def _merge_input_ids_with_image_features(
        self,
        image_features:torch.Tensor, 
        inputs_embeds:torch.Tensor,
        input_ids:torch.Tensor,
        attention_mask:torch.Tensor,
        kv_cache:Optional[KVCache]=None,
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_len = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device


            #[batch_size, seqlen, embed_dim]
        scaled_image_features = image_features/(self.config.hidden_size**0.5)
        final_embeddings = torch.zeros(batch_size,sequence_len, embed_dim, dtype= dtype, device=device)
            
        #[batch_size,seqlen]
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = input_ids == self.config.image_token_index


        pad_mask = input_ids == self.pad_token_id
        

        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1,-1,embed_dim)

        #we did expand mask so that we could use them here in torch.where
        final_embeddings = torch.where(text_mask_expanded,inputs_embeds,final_embeddings)

        # we cannot use torch.where here because seqlen of scaled_image_features != final_embeddings

        final_embeddings = final_embeddings.masked_scatter(image_mask_expanded, scaled_image_features)

        final_embeddings = torch.where(pad_mask_expanded, torch.zeros_like(final_embeddings), final_embeddings)
        min_dtype = torch.finfo(dtype).min
        qlen = inputs_embeds.shape[1]

        if kv_cache == None or kv_cache.num_items() ==0:
            # mask which is added to QK^T before softmax  -> non_attending = -inf but not in case of gemma (0 here)
            casual_mask = torch.full(
                (batch_size,qlen, qlen), fill_value=0, dtype=dtype, device = device
            )
        else:
            # assert qlen==1
            kv_len = kv_cache.num_items()+qlen
             # we again do not need to add mask here because we are using kv_caching hence no need of causal_mask during inference

            casual_mask = torch.full((batch_size,qlen, kv_len), fill_value=0, dtype=dtype, device = device)

        #add the head_dim
        # [batch_size, qlen,kv_len]->[batch_size, num_heads_q, qlen, kv_len] (query only contain 1 token hence only 1 head)
        casual_mask = casual_mask.unsqueeze(1)

        #for rotary position_ids
        if kv_cache is not None and kv_cache.num_items()>0:
            #position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:,-1]
            
            if position_ids.dim() ==1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill((attention_mask==0),1).to(device)

        return final_embeddings, casual_mask, position_ids


    
    def forward(self,input_ids:torch.LongTensor = None, pixel_values:torch.FloatTensor = None, attention_mask:Optional[torch.Tensor]=None, kv_cache: Optional[KVCache]= None)->Tuple:

        assert torch.all(attention_mask == 1), "Input cannot be padded"

        # shape: (Batch_Size, Seq_Len, Hidden_Size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        # batch,channel,height,width = batch,num_patches,embed_dim
        selected_image_features = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        #batch num_patches embed_dim -> batch, num_patches, hidden_size
        image_features = self.multi_modal_projector(selected_image_features)   # basically a linear layer to change vision embed dim to hidden_size of language_model
        # merge embeddings of image and text tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        
        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            kv_cache = kv_cache,
        )

        return outputs



