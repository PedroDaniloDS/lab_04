import numpy as np

### primeiro: base

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + 1e-6)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
        
    weights = softmax(scores)
    return np.matmul(weights, V)

def feed_forward_network(x):
    d_model = x.shape[-1]
    W1 = np.random.randn(d_model, 2048) * 0.01 # colocanod para 2048
    W2 = np.random.randn(2048, d_model) * 0.01
    
    hidden = np.maximum(0, np.matmul(x, W1)) # ativando a relu
    return np.matmul(hidden, W2)

def add_and_norm(x, sublayer_output):
    return layer_norm(x + sublayer_output) # Output = LayerNorm(x + Sublayer(x))


### segundo: encoder

class EncoderBlock:
    def forward(self, x):
        # 1. Self-Attention
        attention_out = scaled_dot_product_attention(x, x, x)
        # 2. Add & Norm
        x = add_and_norm(x, attention_out)
        
        # 3. FFN
        ffn_out = feed_forward_network(x)
        # 4. Add & Norm
        out = add_and_norm(x, ffn_out)
        
        return out # matriz de memória rica, no caso o z


### terceiro: decoder

class DecoderBlock:
    def forward(self, y, Z):
        # 1. Masked Self-Attention (simulação da máscara zerando o futuro)
        seq_len = y.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len))) 
        masked_attention_out = scaled_dot_product_attention(y, y, y, mask=causal_mask)
        
        # 2. Add & Norm 
        y = add_and_norm(y, masked_attention_out)
        
        # 3. Cross-Attention (Q vem de y; K e V vêm de Z) 
        cross_attention_out = scaled_dot_product_attention(Q=y, K=Z, V=Z)
        
        # 4. Add & Norm 
        y = add_and_norm(y, cross_attention_out)
        
        # 5. FFN e último Add & Norm 
        ffn_out = feed_forward_network(y)
        out = add_and_norm(y, ffn_out)
        
        return out

def projecao_linear_e_softmax(x, vocab_size=100):
    W_proj = np.random.randn(x.shape[-1], vocab_size) * 0.01
    logits = np.matmul(x, W_proj)
    probabilidades = softmax(logits)
    
    # Retorna o índice da palavra com maior probabilidade
    return np.argmax(probabilidades[-1])


### quarto: teste de inferência 

def inferencia_autoregressiva():
    encoder = EncoderBlock()
    decoder = DecoderBlock()
    
    # teste basico
    vocab = {0: "<START>", 1: "<EOS>", 2: "Thinking", 3: "Machines", 4: "I", 5: "am", 6: "alive"}
    reverse_vocab = {v: k for k, v in vocab.items()}
    d_model = 512
    
    encoder_input_tokens = [reverse_vocab["Thinking"], reverse_vocab["Machines"]]
    X = np.random.randn(len(encoder_input_tokens), d_model)
    
    Z = encoder.forward(X)
    
    # iniciando o token
    decoder_input_tokens = [reverse_vocab["<START>"]]
    palavras_geradas = ["<START>"]
    
    print("começo da geração")
    
    while True:
        Y = np.random.randn(len(decoder_input_tokens), d_model)
        
        out = decoder.forward(Y, Z)
        
        next_word_idx = projecao_linear_e_softmax(out, vocab_size=len(vocab))
        next_word = vocab.get(next_word_idx, "<UNK>")
        
        decoder_input_tokens.append(next_word_idx)
        palavras_geradas.append(next_word)
        
        if next_word == "<EOS>" or len(palavras_geradas) > 8:
            if next_word != "<EOS>":
                palavras_geradas.append("<EOS>") 
            break
            
    return palavras_geradas

resultado = inferencia_autoregressiva()
print("geracao final:", " ".join(resultado))