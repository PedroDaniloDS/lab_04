# laboratorio 04

esse repositorio contem a entrega da arquitetura encoder-decoder para o lab 04

Devido a complexidade, utilizei inteligência artificial como suporte para estruturar o codigo base em numpy e auxiliar na logica matematica das etapas mais difíceis (aproximadamente metade do trabalho). A IA me ajudou especificamente com:
- O calculo matricial e a aplicacao da mascara causal na função `scaled_dot_product_attention`
- O fluxo correto de tensores passando pelas camadas de `add_and_norm`, ffn e cross-attention nos metodos `forward` do `EncoderBlock` e `DecoderBlock`
- A montagem do laco auto-regressivo de inferencia na funcao `inferencia_autoregressiva()`.

