import torch
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import torch.nn as nn
import os

class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, n_segments, d_model, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        
        if (x < 0).any() or (x >= self.tok_embed.num_embeddings).any():
            raise ValueError("input_ids contains invalid indices.")
        
        if (seg < 0).any() or (seg >= self.seg_embed.num_embeddings).any():
            raise ValueError("segment_ids contains invalid indices.")

        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(self.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)
    
def get_attn_pad_mask(seq_q, seq_k, device, attention_mask=None):
    if attention_mask is not None:
        pad_attn_mask = attention_mask.unsqueeze(1).expand(-1, seq_q.size(1), -1).to(device)
        pad_attn_mask = ~pad_attn_mask.bool()
    else:
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).to(device)
        pad_attn_mask = pad_attn_mask.expand(seq_q.size(0), seq_q.size(1), seq_k.size(1))
    return pad_attn_mask

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, device)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, device):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(device)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        scores.masked_fill_(attn_mask, -1e9) 
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, device):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(self.d_k, self.device)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = nn.Linear(self.n_heads * self.d_v, self.d_model, device=self.device)(context)
        return nn.LayerNorm(self.d_model, device=self.device)(output + residual), attn
    
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
    
class BERT(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_ff, d_k, n_segments, vocab_size, max_len, device):
        super(BERT, self).__init__()
        self.params = {
            'n_layers': n_layers, 'n_heads': n_heads, 'd_model': d_model,
            'd_ff': d_ff, 'd_k': d_k, 'n_segments': n_segments,
            'vocab_size': vocab_size, 'max_len': max_len
        }
        self.embedding = Embedding(vocab_size, max_len, n_segments, d_model, device)
        self.layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, d_k, device) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        self.device = device

    def forward(self, input_ids, segment_ids=None, masked_pos=None, attention_mask=None):
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids).to(self.device)
        if attention_mask is None:
            attention_mask = (input_ids != 0).float().to(self.device)
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device, attention_mask)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        h_pooled = self.activ(self.fc(output[:, 0]))
        logits_nsp = self.classifier(h_pooled)
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
            h_masked = torch.gather(output, 1, masked_pos)
            h_masked = self.norm(F.gelu(self.linear(h_masked)))
            logits_lm = self.decoder(h_masked) + self.decoder_bias
        else:
            logits_lm = None
        return output, logits_lm, logits_nsp

    def get_last_hidden_state(self, input_ids, segment_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != 0).float().to(self.device)
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device, attention_mask)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        return output
    
class TextSimilarityModel:
    def __init__(self, model_path, classifier_head_path, tokenizer_path, device):
        # Load model and tokenizer
        self.device = device
        self.model = self.load_model(model_path)
        self.classifier_head = self.load_classifier_head(classifier_head_path)
        self.tokenizer = self.load_tokenizer(tokenizer_path)

    def load_model(self, model_path):
        # Load the model
        model = BERT(
            n_layers=12,
            n_heads=12,
            d_model=768,
            d_ff=3072,
            d_k=64,
            n_segments=2,
            vocab_size=23068,
            max_len=1000,
            device=self.device
        ).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def load_classifier_head(self, classifier_head_path):
        classifier_head = torch.nn.Sequential(
            torch.nn.Linear(768 * 3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 3)
        ).to(self.device)
        classifier_head.load_state_dict(torch.load(classifier_head_path, map_location=torch.device('cpu')))
        classifier_head.eval()
        return classifier_head

    def load_tokenizer(self, tokenizer_path):
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        return tokenizer

    def predict_label(self, premise, hypothesis):
        # Tokenize input
        inputs_a = self.tokenizer(premise, return_tensors='pt', truncation=True, padding=True).to(self.device)
        inputs_b = self.tokenizer(hypothesis, return_tensors='pt', truncation=True, padding=True).to(self.device)
        inputs_ids_a = inputs_a['input_ids']
        attention_a = inputs_a['attention_mask']
        inputs_ids_b = inputs_b['input_ids']
        attention_b = inputs_b['attention_mask']

        # Get embeddings
        u = self.model(inputs_ids_a, attention_mask=attention_a)[0]
        v = self.model(inputs_ids_b, attention_mask=attention_b)[0]

        # Mean pool the embeddings
        u_pool = self.mean_pool(u, attention_a)
        v_pool = self.mean_pool(v, attention_b)

        # Create input for classifier
        x = torch.cat([u_pool, v_pool, torch.abs(u_pool - v_pool)], dim=-1)

        # Get classification logits
        logits = self.classifier_head(x)
        pred_label = torch.argmax(logits, dim=-1).item()

        # Map to label
        label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
        predicted_label = label_map[pred_label]

        return predicted_label

    def mean_pool(self, token_embeds, attention_mask):
        # Apply mean pooling to get the sentence embeddings
        in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
        return pool