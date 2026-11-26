# Exploração vs. aproveitamento (ε-greedy)

> **Parte 6** · Responsável: _(Guilherme Silva Grati Pinto)_ · Código relacionado: `DQN.Policy`

## O que esta seção precisa cobrir

- [x] O dilema exploration vs. exploitation
- [x] Política gulosa e política ε-greedy
- [x] Decaimento linear do ε (valores usados no projeto)
- [x] Por que o ε não vai até zero durante o treino
- [x] Aleatoriedade pura em Haskell: passar o StdGen explicitamente

## Conceito

Em aprendizado por reforço o agente não sabe de antemão quais ações são boas: ele só descobre agindo. Isso cria um dilema. Se ele sempre escolhe a ação que **hoje** parece a melhor (**aproveitamento**, *exploitation*), pode ficar preso numa estratégia medíocre só porque nunca testou as alternativas. Se ele sempre escolhe ações ao acaso (**exploração**, *exploration*), descobre muita coisa, mas nunca usa o que aprendeu. É como escolher onde almoçar: ir sempre ao restaurante de que você já gosta garante uma refeição boa, mas você nunca vai saber se existe um melhor na rua ao lado.

A **política gulosa** (*greedy*) resolve só metade do problema: em cada estado escolhe a ação de maior Q-valor estimado pela rede. No começo do treino, esses Q-valores são quase aleatórios (a rede acabou de ser inicializada), então agir de forma gulosa é agir com base em lixo, e as experiências guardadas no replay buffer ficam todas parecidas.

A **política ε-greedy** mistura as duas coisas com um único parâmetro $\varepsilon \in [0, 1]$: a cada passo, com probabilidade $\varepsilon$ o agente explora (ação aleatória) e com probabilidade $1 - \varepsilon$ ele aproveita (ação gulosa).

O valor de $\varepsilon$ muda ao longo do treino. No início a rede não sabe nada, então queremos explorar bastante ($\varepsilon$ alto). Conforme ela aprende, faz mais sentido confiar nela ($\varepsilon$ baixo). Por isso usamos um **decaimento linear**: $\varepsilon$ cai em linha reta de um valor inicial até um valor mínimo e depois fica parado nesse mínimo.

**Por que o ε não vai até zero?** Por três motivos:

1. A rede é só uma aproximação e pode estar errada em estados que ela quase não viu. Um pouco de exploração permanente continua testando ações que a rede acha ruins e corrige estimativas que ficaram erradas.
2. O replay buffer precisa de experiências variadas. Com $\varepsilon = 0$ o agente só gera dados da própria política atual, e o buffer fica cheio de situações muito parecidas.
3. A política muda enquanto a rede aprende. Se a exploração parar cedo demais, uma estimativa inicial ruim pode nunca mais ser corrigida.

(No artigo do DQN, Mnih et al. 2015, o ε também decai linearmente e para num valor mínimo maior que zero.) Quando queremos medir o desempenho final, aí sim usamos a política totalmente gulosa ($\varepsilon = 0$), sem treinar.

**Decaimento por passo, não por episódio.** No nosso projeto o ε depende do número total de **passos** de treino. Os episódios têm tamanhos muito diferentes (no começo a haste cai em poucas dezenas de passos, depois pode chegar a 500), então decair por episódio faria o cronograma de exploração depender de quão bem o agente está indo. Contar passos dá um cronograma uniforme em quantidade de experiência.

**Aleatoriedade em Haskell.** Haskell é puro: uma função com as mesmas entradas devolve sempre a mesma saída. Por isso não existe "um número aleatório global". O gerador `StdGen` é um valor que a gente passa para a função, e a função devolve o resultado **junto com o gerador novo**. Isso mantém o código puro e deixa o treino reprodutível: com a mesma semente (`mkStdGen 42`) o programa faz exatamente a mesma coisa.

## Fórmulas

**Política gulosa:**

$$a^{*} = \arg\max_{a} Q(s, a; \theta)$$

**Política ε-greedy.** Com $|\mathcal{A}|$ ações, a probabilidade de escolher cada ação é:

$$
\pi(a \mid s) =
\begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}|} & \text{se } a = a^{*} \\[2mm]
\dfrac{\varepsilon}{|\mathcal{A}|} & \text{caso contrário}
\end{cases}
$$

(a ação gulosa também pode sair no sorteio da exploração, por isso o termo $\varepsilon / |\mathcal{A}|$ aparece nela).

**Decaimento linear do ε**, com $t$ = passos de treino já dados:

$$
\varepsilon(t) =
\begin{cases}
\varepsilon_{\text{start}} + (\varepsilon_{\text{end}} - \varepsilon_{\text{start}}) \cdot \dfrac{t}{T} & \text{se } t < T \\[2mm]
\varepsilon_{\text{end}} & \text{se } t \ge T
\end{cases}
$$

Símbolos:

- $\varepsilon_{\text{start}}$ = `hpEpsilonStart` = 1,0 (no começo, só exploração);
- $\varepsilon_{\text{end}}$ = `hpEpsilonEnd` = 0,05 (mínimo: 5 % de exploração para sempre);
- $T$ = `hpEpsilonDecaySteps` = 10 000 passos;
- $t$ = número de passos (`agentTotalSteps`), não de episódios.

## Exemplo

Com os valores padrão ($\varepsilon_{\text{start}} = 1{,}0$, $\varepsilon_{\text{end}} = 0{,}05$, $T = 10\,000$):

| passo $t$ | cálculo | $\varepsilon(t)$ |
|---|---|---|
| 0 | $1{,}0$ | 1,0 |
| 5 000 | $1{,}0 + (0{,}05 - 1{,}0) \cdot 0{,}5 = 1{,}0 - 0{,}475$ | 0,525 |
| 10 000 | $t \ge T$ | 0,05 |
| 20 000 | $t \ge T$ | 0,05 |

Agora escolha uma ação no passo $t = 5\,000$ ($\varepsilon = 0{,}525$), num estado em que a rede dá $Q = [0{,}8,\ 1{,}3]$ (índice 0 = `PushLeft`, índice 1 = `PushRight`). A ação gulosa é `PushRight`, porque $1{,}3 > 0{,}8$.

- Se o sorteio der $u = 0{,}30$: como $0{,}30 < 0{,}525$, **explora**. Um segundo sorteio escolhe uma ação ao acaso (digamos, índice 0) e o agente faz `PushLeft`, mesmo a rede preferindo `PushRight`.
- Se o sorteio der $u = 0{,}80$: como $0{,}80 \ge 0{,}525$, **aproveita** e faz `PushRight`.

Probabilidades neste passo: $P(\text{PushLeft}) = 0{,}525 / 2 = 0{,}2625$ e $P(\text{PushRight}) = 1 - 0{,}2625 = 0{,}7375$.

## Como isso aparece no nosso código

Tudo está em `src/DQN/Policy.hs`, que exporta três funções:

- **`epsilonAt :: Hyperparams -> Int -> Double`** implementa a fórmula do decaimento linear. Lê `hpEpsilonStart`, `hpEpsilonEnd` e `hpEpsilonDecaySteps` de `DQN.Types`. Para $t \ge T$ devolve `hpEpsilonEnd`, e é isso que impede o ε de ir a zero. Se `hpEpsilonDecaySteps` for 0, devolve `hpEpsilonEnd` direto, para não dividir por zero.
- **`greedyAction :: Network -> Vector -> Action`** implementa $\arg\max_a Q(s,a)$ como `indexToAction (argmax (forward rede s))`. `forward` vem de `DQN.Network` (Parte 3), `argmax` de `DQN.Linalg` (Parte 2, empate pelo menor índice) e `indexToAction` de `DQN.Types` (índice 0 = `PushLeft`, 1 = `PushRight`).
- **`epsilonGreedy :: Double -> Network -> Vector -> StdGen -> (Action, StdGen)`** implementa a política ε-greedy. Sorteia $u$ com `randomR (0, 1)`. Se $u < \varepsilon$, faz um **segundo** sorteio com o gerador novo, `randomR (0, numActions - 1)`, e converte o índice com `indexToAction`. Senão, devolve `greedyAction`. Em ambos os casos devolve o `StdGen` avançado, que é como a aleatoriedade passa de um passo para o outro sem `IO`.

Ligação com o resto do projeto: no `runEpisode` (`DQN.Agent`, Parte 7), a cada passo o agente calcula $\varepsilon$ com `epsilonAt hp agentTotalSteps` e escolhe a ação com `epsilonGreedy` usando a rede **online**. O `EpisodeLog` guarda o ε do fim do episódio (`logEpsilon`). A avaliação final (`evaluatePolicy`, Parte 8) usa a política gulosa pura, sem exploração.

## Perguntas prováveis da arguição

1. **Pergunta:** Por que o ε não vai até zero durante o treino?
   **Resposta:** Porque a rede é uma aproximação e pode estar errada em estados pouco vistos. Um ε mínimo de 0,05 mantém uma exploração permanente, que corrige estimativas ruins e mantém o replay buffer com experiências variadas. Com ε = 0 o agente só reforçaria o que a rede já acredita, certo ou errado. No código, é o ramo `t >= decaySteps` de `epsilonAt` que fixa o valor em `hpEpsilonEnd`.

2. **Pergunta:** Por que o ε decai por passo e não por episódio?
   **Resposta:** Porque os episódios têm durações muito diferentes: no início duram poucas dezenas de passos e depois podem chegar a 500. Decair por episódio faria o cronograma de exploração depender do desempenho do agente. Contando passos (`agentTotalSteps`), o ε depende só da quantidade de experiência. Por isso `epsilonAt` recebe o número de passos.

3. **Pergunta:** Por que `epsilonGreedy` faz dois sorteios em vez de reusar o $u$ para escolher a ação aleatória?
   **Resposta:** Porque, dado que $u < \varepsilon$, o valor de $u$ não é mais uniforme em $[0, 1)$: ele fica restrito a $[0, \varepsilon)$, e com ε pequeno quase sempre cairia na mesma ação. Um segundo sorteio independente, com o gerador novo, garante que a ação explorada seja uniforme entre as ações. Reusar o mesmo $u$ é um erro comum.

4. **Pergunta:** Com 2 ações e ε = 0,2, qual é a probabilidade de o agente escolher a ação que NÃO é a gulosa?
   **Resposta:** 0,1, ou seja, $\varepsilon / 2$. A exploração acontece com probabilidade 0,2, mas a ação aleatória sorteada coincide com a gulosa em metade das vezes. Por isso a gulosa tem probabilidade $1 - 0{,}2 + 0{,}1 = 0{,}9$.

5. **Pergunta:** Por que `epsilonGreedy` recebe e devolve um `StdGen` em vez de usar `randomIO`?
   **Resposta:** Porque o módulo é puro: sem `IO`, a função com as mesmas entradas devolve sempre o mesmo resultado. Passar o gerador explicitamente torna o treino reprodutível (mesma semente, mesmo resultado) e fácil de testar. Se usássemos o mesmo `StdGen` em todos os sorteios sem avançá-lo, sairia sempre o mesmo número.

## Referências

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Sutton, R. S.; Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. 2ª ed. MIT Press.
