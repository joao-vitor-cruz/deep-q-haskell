# Arquitetura do DQN e target network

> **Parte 7** · Responsável: Fagundes · Código relacionado: `DQN.Agent`

## O algoritmo DQN

O agente observa quatro valores contínuos — posição e velocidade do carrinho,
ângulo e velocidade angular da haste — e escolhe entre empurrar o carrinho
para a esquerda ou para a direita. A rede neural aproxima a função `Q(s, a)`,
que estima o retorno futuro esperado ao escolher a ação `a` no estado `s`.
Esse retorno considera as recompensas futuras com o desconto `γ`.

O treinamento combina as seguintes partes anteriores:

1. A política ε-greedy escolhe uma ação usando a rede online. Com probabilidade
   ε, escolhe uma ação aleatória para explorar; no restante dos casos, escolhe
   a ação com o maior Q-valor estimado para o estado atual.
2. O ambiente aplica a ação e devolve o próximo estado, a recompensa e a
   indicação de se o episódio terminou.
3. O agente guarda a transição `(s, a, r, s', done)` no replay buffer.
4. Depois do warmup, sorteia um minibatch do buffer e ajusta a rede online
   para aproximar a previsão `Q(s, a)` do alvo de Bellman calculado para cada
   transição.
5. A cada quantidade configurada de passos, copia a rede online para a rede
   target.

Esse ciclo se repete até o episódio terminar ou atingir o limite de passos.
O treino completo executa a quantidade configurada de episódios.

## Rede online e rede target

A **rede online** prevê os Q-valores das ações possíveis no estado atual do
CartPole. O agente usa essas previsões para escolher uma ação com maior retorno
futuro esperado, embora a política ε-greedy também possa escolher uma ação
aleatória durante a exploração. Depois do warmup (`hpWarmupSteps`), os pesos
da online são atualizados em cada passo de treino (`trainStep`).

A **rede target** é uma cópia separada da online. Para cada transição, ela
estima os Q-valores das ações possíveis no próximo estado `s'`; o maior deles
ajuda a calcular o alvo de Bellman. A target não é o objetivo final do agente:
é usada para calcular um alvo de treino `y` para cada transição. Se a transição
não for terminal, `y` também inclui a recompensa imediata `r` e o desconto `γ`.

A target fica parada entre sincronizações para que o alvo de treino não mude a
cada atualização da online. Depois de processar um passo do ambiente, se o
total de passos atingir um múltiplo de `hpTargetUpdateEvery`, seus pesos são
substituídos por uma cópia dos pesos atuais da online. Manter o alvo estável
entre essas cópias ajuda a reduzir a instabilidade do treinamento.

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

O **warmup** impede o treino antes de o replay buffer reunir a quantidade
mínima de experiências definida por `hpWarmupSteps`. Quando esse limite é
alcançado, cada passo pode sortear um minibatch (`hpBatchSize`) e atualizar a
online.

A target é sincronizada depois de processar o passo cujo total é múltiplo de
`hpTargetUpdateEvery`. Se houve treino nesse passo, a cópia inclui a atualização
mais recente da online. Entre cópias, os alvos de Bellman são calculados com os
mesmos pesos da target.

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
   minibatches. Assim, o primeiro treino só começa depois que o buffer acumulou
   a quantidade mínima de experiências configurada.

4. **Pergunta:** Quando a target é sincronizada?
   **Resposta:** Depois de processar um passo cujo total é múltiplo de
   `hpTargetUpdateEvery`; nesse momento, `agentTarget` recebe uma cópia da
   rede online, incluindo a atualização daquele passo, se houve treino.

5. **Pergunta:** Por que guardar `agentBest` se já temos `agentOnline`?
   **Resposta:** O desempenho do DQN pode oscilar. `agentBest` preserva a rede
   que atingiu a melhor média móvel de recompensa, mesmo que a rede online
   piore em episódios posteriores.

## Referências

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Sutton, R. S.; Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. 2ª ed. MIT Press.
