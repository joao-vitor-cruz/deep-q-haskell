# 09 — Otimizacao, Limitacoes e Aplicacoes do DQN

## 1. Gradiente Descendente (SGD) vs. Adam

Na aproximacao de funcao com redes neurais em Deep Q-Learning (DQN), o objetivo em cada passo de treino e ajustar os parametros (pesos e biases) da rede online para reduzir o erro entre a previsao atual e o alvo de Bellman.

### Gradiente Descendente Simples (SGD)

No gradiente descendente padrao (stochastic gradient descent), cada parametro $w$ e atualizado na direcao oposta ao gradiente $g = \frac{\partial \mathcal{L}}{\partial w}$ multiplicado por uma taxa de aprendizado $\text{lr}$:

$$w' = w - \text{lr} \cdot g$$

No nosso projeto, essa regra e implementada em `src/DQN/Optimizer.hs` pela funcao `sgdStep`, que usa `zipNetworkWith` (de `DQN.Linalg`) para aplicar `\w g -> w - lr * g` em todos os pesos e biases da rede (`Network`).

A principal vantagem do SGD e a simplicidade e o baixo custo por passo. Porem, ele usa uma unica taxa de aprendizado fixa para todos os parametros e nao acumula historico de variacoes passadas. Em problemas de reforco com redes neurais, onde os alvos mudam ao longo do tempo e a escala dos gradientes pode variar bastante entre camadas e iteracoes, o SGD puro tende a oscilar ou progredir lentamente.

### Otimizador Adam

O **Adam** (*Adaptive Moment Estimation*, Kingma e Ba, 2014) combina momentos adaptativos para cada parametro individual:

- **Primeiro momento ($m$):** media movel exponencial do gradiente (funciona como um *momentum*, suavizando ruidos e acelerando direcoes consistentes).
- **Segundo momento ($v$):** media movel exponencial do gradiente ao quadrado (mede a magnitude recente das variacoes de cada parametro, permitindo adaptar o passo por coordenada).

Com hiperparametros padrao $\beta_1 = 0.9$, $\beta_2 = 0.999$ e $\varepsilon = 10^{-8}$ (estabilidade numerica para evitar divisao por zero), um passo a partir de $(w, m, v, t)$ com gradiente $g$ segue:

$$t' = t + 1$$

$$m' = \beta_1 \cdot m + (1 - \beta_1) \cdot g$$

$$v' = \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$$

$$\hat{m} = \frac{m'}{1 - \beta_1^{t'}}$$

$$\hat{v} = \frac{v'}{1 - \beta_2^{t'}}$$

$$w' = w - \text{lr} \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}$$

No codigo:
- `initAdam` (em `DQN.Optimizer`) inicializa o `AdamState` (`adamM` e `adamV` criados com `zeroNetwork`, e `adamStepCount = 0`).
- `adamStep` (em `DQN.Optimizer`) calcula `t'`, atualiza `m'` e `v'` com `zipNetworkWith`, aplica a correcao de vies (`1 - beta1 ^ t'` e `1 - beta2 ^ t'`) e atualiza os pesos da rede online.
- `trainStep` (em `DQN.Agent`) chama `adamStep` usando `hpLearningRate` e `agentOptimizer` para atualizar `agentOnline`.

### Exemplo numerico pequeno

Considere um unico parametro com $w = 1.0$, $\text{lr} = 0.01$, estado inicial $m = 0$, $v = 0$, $t = 0$, e gradiente $g = 2.0$ no primeiro passo:

1. $t' = 0 + 1 = 1$.
2. $m' = 0.9 \cdot 0 + 0.1 \cdot 2.0 = 0.2$.
3. $v' = 0.999 \cdot 0 + 0.001 \cdot (2.0)^2 = 0.004$.
4. Correcao de vies:
   - $\hat{m} = \frac{0.2}{1 - 0.9^1} = \frac{0.2}{0.1} = 2.0$.
   - $\hat{v} = \frac{0.004}{1 - 0.999^1} = \frac{0.004}{0.001} = 4.0$.
5. Atualizacao do peso:
   - $w' = 1.0 - 0.01 \cdot \frac{2.0}{\sqrt{4.0} + 10^{-8}} \approx 1.0 - 0.01 \cdot \frac{2.0}{2.0} = 0.99$.

Observe que, no primeiro passo a partir do zero, a correcao de vies faz $\hat{m}$ e $\hat{v}$ recuperarem a escala original de $g$ e $g^2$, fazendo o deslocamento ficar proximo de $\text{lr}$ no sentido oposto ao gradiente.

---

## 2. Vantagens e Limitacoes do DQN

### Vantagens

- **Generalizacao em espacos de estados continuos:** em vez de uma tabela para cada estado (inviavel no CartPole, onde posicao, velocidade, angulo e velocidade angular sao continuos), a rede neural aproxima $Q(s, a)$ e generaliza entre estados semelhantes.
- **Reaproveitamento de experiencias:** com o replay buffer (`DQN.Replay`, usado em `DQN.Agent`), cada transicao pode ser amostrada em minibatches mais de uma vez.
- **Base modular e extensivel:** separar ambiente (`DQN.Env`), rede (`DQN.Network`), perda/gradiente (`DQN.Backprop`), otimizador (`DQN.Optimizer`) e agente (`DQN.Agent`) facilita testar ajustes e variantes.

### Limitacoes

- **Instabilidade no treinamento:** combinar aproximacao de funcao nao linear (rede neural), *bootstrapping* (usar estimativas futuras no alvo de Bellman) e dados correlacionados pode fazer a politica oscilar ou regredir apos um bom desempenho. No projeto, isso e visivel na necessidade de sincronizar a target network periodicamente em `DQN.Agent` e de guardar `agentBest` ao longo do treino.
- **Superestimacao dos Q-valores:** o operador $\max_{a'} Q_{\text{target}}(s', a')$ (usado em `tdTarget` em `DQN.Agent`) tende a escolher acoes com erro de estimacao positivo alto, inflando os valores alvo e podendo enviesar a politica.
- **Custo de amostras (*sample efficiency*):** o agente precisa de muitas interacoes com o ambiente ate que o replay buffer tenha diversidade suficiente e a rede convirja.
- **Apenas espacos de acoes discretos:** o DQN classico calcula um Q-valor para cada acao discreta (`PushLeft` e `PushRight` no CartPole, via `argmax` em `DQN.Linalg` e `greedyAction` em `DQN.Policy`). Ele nao se aplica diretamente a acoes continuas sem adaptacoes.

---

## 3. Variantes do DQN

- **Double DQN:** separa a escolha da melhor acao da avaliacao do seu valor. Em vez de usar a mesma rede para escolher e avaliar o maximo no alvo de Bellman, uma rede seleciona $\arg\max_{a'} Q(s', a')$ e a outra avalia o valor dessa acao, reduzindo a superestimacao.
- **Dueling DQN:** divide a rede em dois ramos a partir das camadas compartilhadas: uma estimativa de valor do estado $V(s)$ e uma funcao de vantagem $A(s, a)$ por acao, combinadas para formar $Q(s, a)$. Isso ajuda quando o valor do estado importa mais do que a escolha especifica da acao.
- **Prioritized Experience Replay (PER):** em vez de sortear transicoes uniformemente (como faz `sampleBatch` em `DQN.Replay`), prioriza transicoes com maior erro TD (maior surpresa/aprendizado) e corrige o vies de amostragem com pesos de importancia.
- **Rainbow:** combina varias melhorias sobre o DQN base (como Double DQN, Prioritized Replay, Dueling DQN, retornos em multi-passos, redes distribucionais e ruido na exploracao), alcancando desempenho muito superior em benchmarks.

---

## 4. Aplicacoes

- **Jogos (como Atari e simulacoes de controle):** o marco inicial do DQN foi aprender a jogar multiplos jogos do Atari diretamente a partir dos pixels da tela; no nosso trabalho, aplicamos o conceito ao controle classico do CartPole.
- **Robotica:** controle de navegacao, manipulacao e tarefas com sensores ruidosos (frequentemente combinando variantes do DQN ou algoritmos de controle continuo quando ha atuadores continuos).
- **Sistemas de recomendacao e operacoes:** selecao sequencial de itens ou decisoes ao longo do tempo buscando maximizar engajamento ou retorno acumulado de longo prazo, respeitando restricoes de latencia e dados.

---

## 5. Perguntas Provaveis da Arguicao e Respostas

1. **Por que usar Adam em vez de SGD puro no DQN?**
   *Resposta:* O SGD puro usa uma taxa de aprendizado fixa igual para todos os parametros e sem momento acumulado, o que costuma tornar o aprendizado lento ou instavel em redes neurais treinadas com alvos nao estacionarios como no DQN. O Adam adapta o passo por parametro usando primeiro e segundo momentos dos gradientes, permitindo convergencia mais estavel no CartPole.

2. **Por que o Adam precisa de correcao de vies ($\hat{m}$ e $\hat{v}$), e o que aconteceria se usassemos $t = 0$ no primeiro passo?**
   *Resposta:* Como os momentos comecam zerados (`initAdam`), nos primeiros passos `m'` e `v'` ficam enviesados para zero; dividir por $1 - \beta_1^{t'}$ e $1 - \beta_2^{t'}$ corrige esse efeito. Usar $t = 0$ na primeira atualizacao faria $1 - \beta^0 = 0$, causando divisao por zero.

3. **Qual e a diferenca entre o $\varepsilon$ do Adam e o $\varepsilon$ do `epsilonGreedy`?**
   *Resposta:* No Adam (`adamStep`), $\varepsilon = 10^{-8}$ e uma constante pequena de estabilidade numerica somada ao denominador $\sqrt{\hat{v}}$ para evitar divisao por zero. Ja na politica $\varepsilon$-greedy (`DQN.Policy`), $\varepsilon$ e a probabilidade de explorar escolhendo uma acao aleatoria em vez da acao gulosa de maior Q-valor.

4. **Como `adamStep` usa `zipNetworkWith` da Parte 2?**
   *Resposta:* `zipNetworkWith` aplica uma funcao binaria elemento a elemento em todos os pesos e biases de duas redes. Em `adamStep`, usamos `zipNetworkWith` para calcular `m'` e `v'`, depois para calcular o passo por parametro a partir de `m'` e `v'`, e por fim para subtrair esse passo da rede atual.

5. **Por que o DQN pode piorar depois de ja ter atingido uma boa pontuacao, e como o projeto lida com isso?**
   *Resposta:* Devido a instabilidade do treinamento com aproximacao de funcao e *bootstrapping* (alvos que mudam durante o treino). Em `DQN.Agent`, o agente sincroniza a target network periodicamente e o `train` acompanha a media movel das recompensas na janela `hpBestWindow`, salvando a rede em `agentBest` quando ha melhora.