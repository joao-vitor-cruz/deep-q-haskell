# Arquitetura do DQN e target network

> **Parte 7** · Responsável: Fagundes · Código relacionado: `DQN.Agent`

## O algoritmo DQN

No CartPole, o agente observa quatro valores contínuos — posição e velocidade
do carrinho, ângulo e velocidade angular da haste — e escolhe entre empurrar
para a esquerda ou para a direita. A rede neural aproxima a função
`Q(s, a)`: quanto retorno futuro se espera ao escolher a ação `a` no estado
`s`.

O treinamento combina as peças das partes anteriores:

1. A política ε-greedy escolhe uma ação usando a rede online. Com probabilidade
   ε, escolhe uma ação aleatória para explorar; no restante dos casos, escolhe
   a ação com maior Q-valor para aproveitar o que já aprendeu.
2. O ambiente aplica a ação e devolve o próximo estado, a recompensa e a
   indicação de término.
3. O agente guarda a transição `(s, a, r, s', done)` no replay buffer.
4. Depois do warmup, sorteia um minibatch do buffer e ajusta a rede online
   para aproximar seus Q-valores dos alvos de Bellman.
5. A cada quantidade configurada de passos, copia a rede online para a rede
   target.

Esse ciclo se repete até o episódio terminar ou atingir o limite de passos.
O treino completo executa a quantidade configurada de episódios.

## Rede online e rede target

A **rede online** é a rede que escolhe as ações e recebe as atualizações do
gradiente. A **rede target** é uma cópia dos pesos da online, mas fica parada
entre sincronizações. Ela calcula o alvo usado para treinar a online.

Se usássemos a mesma rede para prever o Q-valor atual e para calcular o alvo,
ambos mudariam a cada atualização. Isso faria o objetivo de treino se mover
junto com a previsão, aumentando a instabilidade. A target reduz esse problema:
seus pesos permanecem fixos por vários passos e só são substituídos pelos pesos
da online a cada `hpTargetUpdateEvery` passos.

## Fórmula do alvo de Bellman

Para uma transição `(s, a, r, s', done)`, o alvo é:

$$
y =
\begin{cases}
r, & \text{se } done = \text{True} \\
r + \gamma \max_{a'} Q_{\text{target}}(s', a'), & \text{se } done = \text{False}
\end{cases}
$$

`r` é a recompensa observada, `s'` é o próximo estado, `γ` (`hpGamma`) é o
fator de desconto e `a'` percorre as ações possíveis. Se o episódio terminou,
não existe retorno futuro a estimar: por isso o alvo é somente `r`. Caso
contrário, somamos à recompensa o maior Q-valor futuro previsto pela rede
target, descontado por `γ`.

A rede online é treinada para aproximar esse alvo na ação que realmente foi
escolhida. O código usa a perda de Huber e Adam, implementados nas partes
anteriores.

## Exemplo numérico

Suponha que o agente receba recompensa `r = 1`, use `γ = 0,9` e que, no estado
seguinte, a rede target estime os Q-valores `[2, 3]`. Como a transição não é
terminal:

$$y = 1 + 0,9 \times \max(2, 3) = 3,7.$$

A rede online é então ajustada para que o Q-valor da ação escolhida se aproxime
de `3,7`. Se a transição fosse terminal, o alvo seria `y = 1`; não somaríamos
nenhum valor do próximo estado.

## Warmup, sincronização e melhor rede

O **warmup** impede o treino antes de haver experiências suficientes no replay
buffer (`hpWarmupSteps`). Quando o limite é alcançado, cada passo pode sortear
um minibatch (`hpBatchSize`) e fazer uma atualização da online.

A target é sincronizada depois do treino quando o total de passos é múltiplo de
`hpTargetUpdateEvery`. Assim, durante os passos entre cópias, os alvos de
Bellman são calculados com os mesmos pesos.

O DQN pode oscilar: uma rede que teve bom desempenho pode piorar mais tarde.
Por isso, depois de cada episódio, `train` calcula a média de recompensa da
última janela de `hpBestWindow` episódios. Se essa média superar
`agentBestScore`, guarda uma cópia da rede online em `agentBest`. Isso preserva
a melhor rede encontrada, em vez de assumir que a rede do último episódio será
sempre a melhor.

## Como isso aparece no código

O módulo `DQN.Agent` liga as partes do projeto:

- `newAgent` inicializa a rede online e cria cópias iniciais para `agentTarget`
  e `agentBest`, além de preparar o buffer, o Adam e o contador de passos.
- `tdTarget` aplica a fórmula de Bellman e não faz bootstrap em transições
  terminais.
- `trainStep` calcula alvos com `agentTarget`, calcula o gradiente da rede
  online e atualiza somente a online e o estado do Adam.
- `runEpisode` escolhe ações, interage com o ambiente, armazena transições,
  respeita o warmup, treina e sincroniza a target nos intervalos configurados.
  Também produz o log do episódio.
- `train` repete os episódios, mantém os logs em ordem e atualiza
  `agentBest` quando a média móvel melhora.

## Perguntas prováveis da arguição

1. **Pergunta:** Por que o alvo de uma transição terminal não inclui
   `γ * max Q(s', a')`?
   **Resposta:** Porque o episódio acabou e não há recompensa futura depois
   daquela transição. O alvo correto é apenas a recompensa observada `r`.

2. **Pergunta:** Qual é a diferença entre a rede online e a target?
   **Resposta:** A online escolhe ações e é atualizada a cada passo de treino.
   A target calcula os alvos de Bellman e fica congelada entre sincronizações,
   o que reduz a mudança contínua do objetivo de treino.

3. **Pergunta:** O que o warmup faz?
   **Resposta:** Aguarda o buffer atingir `hpWarmupSteps` antes de sortear
   minibatches. Assim, o primeiro treino usa experiências variadas em vez de
   depender de uma memória quase vazia.

4. **Pergunta:** Quando a target é sincronizada?
   **Resposta:** Quando o total de passos do agente é múltiplo de
   `hpTargetUpdateEvery`; nesse momento, `agentTarget` recebe uma cópia da
   rede online atualizada.

5. **Pergunta:** Por que guardar `agentBest` se já temos `agentOnline`?
   **Resposta:** O desempenho do DQN pode oscilar. `agentBest` preserva a rede
   que atingiu a melhor média móvel de recompensa, mesmo que a rede online
   piore em episódios posteriores.

## Referências

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Sutton, R. S.; Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. 2ª ed. MIT Press.
