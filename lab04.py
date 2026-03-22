import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

# Importando a arquitetura do seu arquivo lab04.py
# (Garante a modularização pedida no documento)
from lab04 import SeuTransformer 

#### tarefas 1 e 2: dataset(hugging) e tokenizar

dataset = load_dataset("bentrevett/multi30k", split="train[:1000]") 
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased") 

def preprocessamento_de_dados(exemplos):
    src_texts = exemplos['en']
    trg_texts = exemplos['de']
    
    src_tokens = tokenizer(src_texts, padding="max_length", truncation=True, max_length=20, return_tensors="pt")
    trg_tokens = tokenizer(trg_texts, padding="max_length", truncation=True, max_length=20, return_tensors="pt")
    
    return {
        "src_ids": src_tokens["input_ids"],
        "trg_ids": trg_tokens["input_ids"]
    }

dados_processados = preprocessamento_de_dados(dataset)

#### tarefa 3: training loop

# Instanciando o modelo importado do lab04
modelo = SeuTransformer(d_model=128, vocab_size=tokenizer.vocab_size)

otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001) 
pad_idx = tokenizer.pad_token_id
criterio_perda = nn.CrossEntropyLoss(ignore_index=pad_idx) 

epocas = 15 
batch_size = 32
n_batches = len(dados_processados["src_ids"]) // batch_size

print("start")

for epoca in range(epocas):
    loss_acumulada = 0
    
    for i in range(n_batches):
        inicio = i * batch_size
        fim = inicio + batch_size
        
        src = dados_processados["src_ids"][inicio:fim]
        trg = dados_processados["trg_ids"][inicio:fim]
        
        trg_input = trg[:, :-1]
        trg_esperado = trg[:, 1:]
        
        otimizador.zero_grad()
        
        saida = modelo(src, trg_input)
        
        saida_reshaped = saida.contiguous().view(-1, saida.shape[-1])
        trg_esperado_reshaped = trg_esperado.contiguous().view(-1)
        
        loss = criterio_perda(saida_reshaped, trg_esperado_reshaped) 
        
        loss.backward() 
        otimizador.step() 
        
        loss_acumulada += loss.item()
        
    media_loss = loss_acumulada / n_batches
    print(f"epoca: {epoca+1} | loss: {media_loss:.4f}") 


#### Tarefa 4: A Prova de Fogo (Overfitting Test)
print("\nIniciando teste de overfitting na frase 0...")
modelo.eval() 

frase_treino_src = dados_processados["src_ids"][0:1]
frase_treino_trg = dados_processados["trg_ids"][0:1]

with torch.no_grad():
    memoria_z = modelo.encoder(modelo.emb(frase_treino_src))
    
    tokens_gerados = [tokenizer.cls_token_id]
    
    for _ in range(20):
        trg_tensor = torch.tensor([tokens_gerados])
        trg_emb = modelo.emb(trg_tensor)
        
        seq_len = trg_tensor.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).unsqueeze(0)
        
        out = modelo.decoder(trg_emb, memoria_z, causal_mask)
        
        next_token_logits = modelo.fc_out(out[:, -1, :])
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        
        tokens_gerados.append(next_token)
        
        if next_token == tokenizer.sep_token_id:
            break

frase_original = tokenizer.decode(frase_treino_trg[0], skip_special_tokens=True)
frase_gerada = tokenizer.decode(tokens_gerados, skip_special_tokens=True)

print(f"Esperado: {frase_original}")
print(f"Gerado:   {frase_gerada}")