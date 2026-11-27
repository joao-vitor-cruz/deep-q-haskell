# Experimento e análise dos resultados

> **Parte 8** · Responsável: Lucas Conceição Ricardo · Código relacionado: `DQN.Experiment` e `app/Main.hs`

## Conceito

Treinar uma rede neural por aprendizado por reforço (*Reinforcement Learning*) exige separar claramente duas fases distintas:

1. **Fase de treinamento (com exploração):** o agente interage com o ambiente utilizando uma política $\varepsilon$-greedy, cometendo ações aleatórias propositalmente para descobrir estados novos e preencher o *Replay Buffer*, enquanto ajusta os pesos da rede online via gradiente descendente (Adam).
2. **Fase de avaliação (puramente gulosa):** após o treino, congelamos os pesos da rede e desligamos a exploração ($\varepsilon = 0$). Nesse modo, o agente sempre escolhe a ação de maior valor Q estimado (`greedyAction`), permitindo medir o real desempenho aprendido sem o ruído de movimentos aleatórios.

Além disso, como o retorno de um único episódio no CartPole varia bastante devido à posição inicial aleatória da haste e às decisões estocásticas durante o treino, analisar episódios isolados gera muito ruído visual. Por isso, aplicamos uma **média móvel** sobre as recompensas para enxergar a tendência real de aprendizado e salvamos um *checkpoint* da **melhor rede** (`agentBest`), protegendo o resultado final contra a instabilidade típica do DQN (fenômeno conhecido como *catastrophic forgetting*, em que a rede aprende a equilibrar a haste, mas piora em episódios posteriores).

## Hiperparâmetros do experimento

Os valores definidos em `defaultHyperparams` (`src/DQN/Types.hs`) controlam o equilíbrio entre velocidade de aprendizado e estabilidade:

| Hiperparâmetro | Valor | Papel no experimento |
|---|---|---|
| `hpLayerSizes` | `[4, 64, 2]` | Arquitetura da rede: 4 entradas do estado, 1 camada oculta com 64 neurônios (ReLU) e 2 saídas lineares (uma por ação). |
| `hpGamma` ($\gamma$) | `0.99` | Fator de desconto; valor próximo de 1 faz o agente valorizar recompensas futuras (manter a haste em pé por muitos passos). |
| `hpLearningRate` ($\alpha$) | `0.001` | Passo de atualização do otimizador Adam; pequeno o suficiente para evitar divergência dos Q-valores. |
| `hpBatchSize` | `32` | Quantidade de transições sorteadas do buffer por passo de treino, reduzindo a variância do gradiente. |
| `hpBufferCapacity` | `10000` | Tamanho máximo da memória FIFO de experiências, quebrando a correlação temporal entre passos consecutivos. |
| `hpWarmupSteps` | `500` | Passos iniciais coletados antes do primeiro treino para garantir diversidade mínima no *Replay Buffer*. |
| `hpTargetUpdateEvery` | `500` | Frequência (em passos) de cópia da rede `online` para a `target`, estabilizando o alvo de Bellman. |
| `hpEpsilonStart` / `End` | `1.0` $\to$ `0.05` | Taxa de exploração inicial (100% aleatório) até o mínimo residual (5% aleatório). |
| `hpEpsilonDecaySteps` | `10000` | Número de passos para o decaimento linear de $\varepsilon$. |
| `hpMaxStepsPerEpisode` | `500` | Teto máximo de passos por episódio (equivalente ao limite do `CartPole-v1`). |
| `hpEpisodes` | `400` | Total de episódios executados no treinamento padrão. |
| `hpBestWindow` | `20` | Tamanho da janela de episódios usada para eleger `agentBest`. |

## Fórmulas

### 1. Média móvel simples com borda inicial parcial

Dada uma sequência de recompensas totais por episódio $R_0, R_1, \dots, R_{N-1}$ e uma janela de tamanho $w \ge 1$, a média móvel no episódio $i$ é:

$$\bar{R}_i = \frac{1}{k} \sum_{j = \max(0,\, i - w + 1)}^{i} R_j, \quad \text{onde } k = \min(w,\, i + 1)$$

Nos primeiros $w - 1$ episódios, em que ainda não existem $w$ amostras anteriores, dividimos pela quantidade $k$ de episódios já observados, preservando o tamanho original da série.

### 2. Avaliação gulosa sem exploração

Durante a avaliação ($\varepsilon = 0$), em cada passo $t$ o agente seleciona determinísticamente a ação:

$$a_t = \arg\max_{a \in \{\text{PushLeft}, \text{PushRight}\}} Q_\theta(s_t, a)$$

O retorno de um episódio de avaliação é a soma das recompensas até o estado terminal ou até atingir $T_{\max} = \text{hpMaxStepsPerEpisode}$:

$$G = \sum_{t=0}^{T - 1} r_t, \quad 1 \le T \le T_{\max}$$

Como o CartPole concede $r_t = 1.0$ a cada passo sobrevivido, $G$ corresponde exatamente ao número de passos em que a haste permaneceu equilibrada.

## Exemplo

Suponha que nos 4 primeiros episódios um agente obteve as recompensas $R = [10, 20, 30, 60]$ e queremos calcular a média móvel com janela $w = 3$:

- **$i = 0$:** janela $[10]$ ($k = 1$) $\implies \bar{R}_0 = \frac{10}{1} = 10.0$
- **$i = 1$:** janela $[10, 20]$ ($k = 2$) $\implies \bar{R}_1 = \frac{10 + 20}{2} = 15.0$
- **$i = 2$:** janela $[10, 20, 30]$ ($k = 3$) $\implies \bar{R}_2 = \frac{10 + 20 + 30}{3} = 20.0$
- **$i = 3$:** janela $[20, 30, 60]$ ($k = 3$) $\implies \bar{R}_3 = \frac{20 + 30 + 60}{3} = 36.67$

Resultado final: $[10.0, 15.0, 20.0, 36.67]$.

## Como isso aparece no nosso código

- **Execução do experimento (`app/Main.hs`):**
  - Lê argumentos opcionais da linha de comando (`cabal run dqn -- <semente> <episodios>`, padrão `42` e `400`).
  - Chama `train` (`DQN.Agent`) para obter o agente treinado e a lista `[EpisodeLog]`.
  - Cria o diretório `results/` e grava `results/training.csv` com os dados de cada episódio.
  - Exibe no terminal o gráfico ASCII da média móvel de 20 episódios e imprime a comparação de desempenho médio (em 20 episódios) entre a **política aleatória** (`evaluateRandomPolicy`, ~20–25 passos), a **rede final** (`agentOnline`) e a **melhor rede** (`agentBest`).
- **Funções puras de análise e visualização (`src/DQN/Experiment.hs`):**
  - `logsToCsv`: serializa `[EpisodeLog]` no formato CSV com o cabeçalho `episode,total_reward,epsilon,avg_loss`, utilizando `printf` (`%.2f`, `%.4f` e `%.6f`) para evitar notação científica em valores pequenos como `epsilon = 0.05` ou perdas baixas.
  - `movingAverage`: implementa exatamente a média móvel com janela parcial nos primeiros elementos.
  - `evaluatePolicy`: roda $n$ episódios usando `resetEnv`, `stepEnv` (`DQN.Env`) e `greedyAction` (`DQN.Policy`) propagando o gerador `StdGen` puramente, sem alterar pesos nem usar $\varepsilon$.
  - `asciiChart`: normaliza os valores entre mínimo e máximo e agrupa intervalos por média (`resample`) quando a quantidade de episódios excede a largura do terminal, desenhando as barras com `#`.

## Perguntas prováveis da arguição

1. **Pergunta:** Por que avaliamos tanto a `agentOnline` (rede final) quanto a `agentBest` (melhor rede), e por que elas podem dar resultados diferentes?
   **Resposta:** Porque o DQN sofre de instabilidade de treinamento: como a função Q é aproximada por uma rede neural e os alvos de Bellman mudam conforme a rede aprende, atualizações tardias podem degradar uma política que já estava boa (*catastrophic forgetting*). Durante o treino (`DQN.Agent.train`), salvamos em `agentBest` uma cópia da rede no instante em que ela atingiu a maior média móvel de 20 episódios. Assim, mesmo que a rede final oscile nos últimos episódios, preservamos o melhor modelo obtido.

2. **Pergunta:** Qual a diferença entre o loop de `runEpisode` (no treino) e o loop de `evaluatePolicy` (na avaliação)?
   **Resposta:** No treino (`DQN.Agent.runEpisode`), o agente usa `epsilonGreedy` (com $\varepsilon > 0$ para explorar), armazena transições no `ReplayBuffer`, atualiza os pesos da rede online via Adam e sincroniza a `target network`. Já em `evaluatePolicy` (`DQN.Experiment`), usamos exclusivamente `greedyAction` ($\varepsilon = 0$), não armazenamos experiências e não alteramos os pesos da rede; o único elemento aleatório é o estado inicial sorteado por `resetEnv`.

3. **Pergunta:** Por que `DQN.Experiment` recebe e devolve `StdGen` em `evaluatePolicy` em vez de usar `randomIO`, e onde fica todo o `IO` do projeto?
   **Resposta:** Para manter o núcleo do projeto 100% puro e determinístico dada uma semente inicial (`mkStdGen seed`), facilitando testes unitários reproduzíveis e evitando efeitos colaterais ocultos. Todo o `IO` (leitura de argumentos do terminal, criação da pasta `results/`, escrita do arquivo CSV e impressão do gráfico) fica isolado em `app/Main.hs`.

## Referências

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Sutton, R. S.; Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. 2ª ed. MIT Press.