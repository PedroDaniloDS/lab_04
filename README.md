# laboratorio 05: Treinamento Fim-a-Fim do Transformer

## Uso de Inteligência Artificial e Adaptações

utilizei IA para agilizar a importação do dataset e a configuração da tokenização (Tarefas 1 e 2). 

Durante o desenvolvimento, tambem recorri a IA para entender e resolver um problema de integraçao de arquitetura. No Lab 04 original eu havia construido o encoder e o decoder como blocos separados, pore, para o Lab 05, percebi que o otimizador do torch precisava acessar todos os parametros da rede de uma so vez. A IA me ajudou a entender que eu precisava criar uma classe unica (`Transformer`) no arquivo `lab04.py` herdando de `nn.Module`. essa classe funcionou como um "embrulho" para unificar as camadas de embedding, encoder, decoder e linear final, permitindo que o `lab05.py` apenas a importasse de forma modular.

Por fim, utilizei a IA como suporte na Tarefa 4 (Teste de Overfitting) para me ajudar a adaptar a logica matematica do laço auto-regressivo do Lab 04 para o formato de tensores exigido pelo PyTorch, garantindo a geração correta token a token.

Partes geradas/complementadas com IA, revisadas por Pedro.