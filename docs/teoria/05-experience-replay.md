# Experience replay

> **Parte 5** · Responsável: _(seu nome completo)_ · Código relacionado: `DQN.Replay`

## O que esta seção precisa cobrir

- [x] O problema de treinar com experiências em sequência (amostras correlacionadas)
- [x] O que é o replay buffer e o que é uma transição (s, a, r, s', done)
- [x] Buffer circular (FIFO) com capacidade máxima
- [x] Amostragem uniforme de minibatches e reaproveitamento de dados
- [x] Por que usamos Data.Sequence em vez de lista em Haskell

## Conceito

Enquanto joga, o agente gera uma experiência a cada passo: estava no estado $s$, escolheu a ação $a$, recebeu a recompensa $r$, foi parar no estado $s'$ e o episódio terminou ou não ($done$). Essa quíntupla $(s, a, r, s', done)$ é uma **transição**.

O jeito mais ingênuo de treinar seria usar cada transição para atualizar a rede logo depois de vivê-la, e depois jogá-la fora. Isso tem dois problemas:

1. **Amostras correlacionadas.** Passos seguidos de um mesmo episódio são quase iguais: no CartPole, o estado no passo $t$ e no passo $t+1$ diferem só por 0,02 s de física. O gradiente descendente supõe que os exemplos de treino são independentes entre si. Se a rede vê 50 exemplos parecidos em sequência, ela "decora" aquela região do espaço de estados e esquece o resto (*esquecimento catastrófico*), e o treino oscila.
2. **Desperdício de dados.** Cada experiência custa um passo no ambiente, mas só seria usada uma vez.

O **experience replay** (Lin, 1992; usado no DQN por Mnih et al., 2015) resolve os dois: o agente guarda as transições numa memória, o **replay buffer**, e, a cada passo de treino, **sorteia** um minibatch aleatório dessa memória. Um minibatch sorteado mistura transições de episódios e momentos diferentes, então as amostras ficam praticamente independentes. E como a transição continua na memória, ela pode ser sorteada várias vezes, em vários passos de treino.

A memória tem **capacidade máxima** (10.000 transições no nosso projeto). Quando enche, a transição **mais antiga** é descartada para a nova entrar: é uma fila **FIFO** (*first in, first out*), também chamada de buffer circular. Descartar as mais antigas faz sentido porque foram geradas por uma política pior (do começo do treino) e porque a memória não pode crescer sem limite.

## Fórmulas

Seja $D = \{e_1, e_2, \dots, e_N\}$ o conteúdo do buffer, com $N \le C$ (capacidade), e cada $e_i = (s_i, a_i, r_i, s'_i, done_i)$.

**Inserção (FIFO):**

$$D \leftarrow \begin{cases} D \cup \{e_{novo}\} & \text{se } N < C \\ (D \setminus \{e_{mais\ antiga}\}) \cup \{e_{novo}\} & \text{se } N = C \end{cases}$$

**Amostragem uniforme com reposição:** para cada um dos $B$ itens do minibatch (no projeto, $B = 32$), sorteia-se um índice

$$i \sim \mathcal{U}\{0, 1, \dots, N-1\}, \qquad P(e_i) = \frac{1}{N}$$

e cada sorteio é independente dos outros (a mesma transição pode sair mais de uma vez).

O minibatch entra na perda do DQN como uma média:

$$L(\theta) = \frac{1}{B} \sum_{(s,a,r,s',done) \in \text{minibatch}} \text{Huber}\big(Q(s, a; \theta),\ y\big)$$

onde $y$ é o alvo de Bellman calculado com a target network (Parte 7).

## Exemplo

Buffer com capacidade $C = 3$, inserindo as transições $e_1, \dots, e_5$:

| Inserção | Conteúdo do buffer (mais antiga → mais nova) | Tamanho |
|---|---|---|
| $e_1$ | $[e_1]$ | 1 |
| $e_2$ | $[e_1, e_2]$ | 2 |
| $e_3$ | $[e_1, e_2, e_3]$ | 3 (cheio) |
| $e_4$ | $[e_2, e_3, e_4]$ ($e_1$ saiu) | 3 |
| $e_5$ | $[e_3, e_4, e_5]$ ($e_2$ saiu) | 3 |

Agora sorteamos um minibatch de $B = 4$. Os índices válidos são $0, 1, 2$ (nunca $3$, que é o tamanho). Suponha que o gerador aleatório dê os índices $2, 0, 2, 1$: o minibatch é $[e_5, e_3, e_5, e_4]$. A transição $e_5$ saiu duas vezes, porque o sorteio é com reposição. Cada sorteio usa o gerador devolvido pelo sorteio anterior; se usássemos sempre o mesmo gerador, sairia sempre o mesmo índice.

## Como isso aparece no nosso código

O tipo do buffer está em `DQN.Types`: `ReplayBuffer` tem a capacidade (`bufferCapacity`) e as transições (`bufferItems :: Seq Transition`). A transição é o tipo `Transition`, com os campos `tState`, `tAction`, `tReward`, `tNextState` e `tDone`.

O módulo `DQN.Replay` implementa:

- **`emptyBuffer capacidade`**: cria o buffer vazio (`Seq.empty`).
- **`pushTransition t buffer`**: é a inserção FIFO. Se `Seq.length items >= capacidade`, remove a mais antiga com `Seq.drop 1` (o começo da sequência) e depois insere a nova no fim com `|>`. Assim o tamanho nunca passa da capacidade.
- **`bufferSize buffer`**: `Seq.length` das transições.
- **`sampleBatch n buffer gen`**: sorteia `n` índices com `randomR (0, tamanho - 1)`, passando o `StdGen` novo de um sorteio para o próximo, pega cada transição com `Seq.index` e devolve `(transições, StdGen final)`. Pré-condição: buffer não vazio.

**Por que `Data.Sequence` e não lista?** Em Haskell, uma lista é encadeada: acessar o elemento $i$ com `!!` custa $O(i)$ e inserir no fim custa $O(n)$. Com 10.000 transições e 32 sorteios por passo de treino, isso seria lento. A `Seq` é uma *finger tree*: inserir no fim (`|>`) e remover do começo (`Seq.drop 1`) custam $O(1)$ amortizado, e `Seq.index` custa $O(\log n)$.

**Por que o `StdGen` é passado explicitamente?** O projeto é puro (sem `IO` fora do `Main`). Uma função pura com a mesma entrada dá sempre a mesma saída, então a aleatoriedade vem de fora: `sampleBatch` recebe um gerador e devolve o gerador novo, que o agente usa no próximo sorteio.

Quem usa o buffer é o `DQN.Agent` (Parte 7): a cada passo, `runEpisode` chama `pushTransition`; quando `bufferSize ≥ hpWarmupSteps` (500), chama `sampleBatch hpBatchSize` e passa o minibatch para `trainStep`. O *warmup* garante que o primeiro minibatch já seja sorteado de uma memória razoavelmente variada.

## Perguntas prováveis da arguição

1. **Pergunta:** Por que não treinar a rede com cada transição logo que ela acontece?
   **Resposta:** Porque transições seguidas são muito parecidas (correlacionadas). O gradiente descendente supõe exemplos independentes; com exemplos em sequência, a rede se ajusta demais à situação atual e esquece o resto, e o treino fica instável. Além disso, cada experiência seria usada uma vez só.

2. **Pergunta:** Quando o buffer enche, qual transição sai e por quê?
   **Resposta:** A mais antiga (FIFO). Ela foi gerada por uma política mais antiga, pior, e a memória precisa ter tamanho limitado. No código, `pushTransition` faz `Seq.drop 1` (remove o começo) antes de `|> t` (insere no fim).

3. **Pergunta:** O sorteio é com ou sem reposição? Isso é um problema?
   **Resposta:** Com reposição: a mesma transição pode sair duas vezes no mesmo minibatch. Com 10.000 transições e minibatch de 32, a chance de repetir é pequena, e o sorteio com reposição é mais simples (cada índice é independente). É o que o artigo do DQN faz.

4. **Pergunta:** Qual o intervalo dos índices sorteados?
   **Resposta:** De 0 a `tamanho − 1`. Sortear até `tamanho` daria um índice fora da sequência e o `Seq.index` falharia.

5. **Pergunta:** Por que usar `Data.Sequence` em vez de lista?
   **Resposta:** Porque precisamos inserir no fim, remover do começo e acessar por índice. Na lista, `!!` é $O(n)$; na `Seq`, `index` é $O(\log n)$ e inserir/remover nas pontas é $O(1)$.

6. **Pergunta:** O que aconteceria se `sampleBatch` usasse o mesmo `StdGen` em todos os sorteios?
   **Resposta:** Como a função é pura, o mesmo gerador dá sempre o mesmo número: o minibatch teria 32 cópias da mesma transição. Por isso cada sorteio usa o gerador devolvido pelo anterior.

7. **Pergunta:** Que variação do replay existe?
   **Resposta:** O *Prioritized Experience Replay* (Schaul et al., 2016), que sorteia com mais frequência as transições com maior erro TD, em vez de uniforme. No nosso projeto usamos o uniforme, como no DQN original.

## Referências

- Lin, L.-J. (1992). *Self-improving reactive agents based on reinforcement learning, planning and teaching*. Machine Learning, 8, 293–321.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Schaul, T. et al. (2016). *Prioritized Experience Replay*. ICLR.
- Sutton, R. S.; Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. 2ª ed. MIT Press.
