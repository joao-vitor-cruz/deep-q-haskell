# AGENTS.md: instruções para qualquer assistente de IA neste projeto

> **Para o integrante do grupo:** se a sua IA não lê este arquivo sozinha
> (ChatGPT ou Gemini no navegador, por exemplo), cole este arquivo inteiro no
> início da conversa e diga qual é a sua parte.
>
> **Para a IA:** leia este arquivo inteiro antes de escrever qualquer código.
> As regras daqui valem mais do que o seu estilo padrão.

---

## 1. Contexto

Trabalho em grupo da faculdade (9 integrantes) sobre **Deep Q-Learning (DQN)**,
implementado **em Haskell puro**. Um agente aprende a equilibrar uma haste em
cima de um carrinho (**CartPole**) usando uma rede neural que aproxima a
função Q.

Como o trabalho é avaliado (isso muda como você deve ajudar):

- **8 dos 10 pontos são arguição oral.** O professor pergunta **qualquer parte
  para qualquer integrante** e quer saber se a pessoa entende o código e as
  decisões de implementação.
- O repositório precisa de **pelo menos 2 commits por integrante**, feitos da
  conta de cada um.

Por isso, a regra mais importante para a IA:

> **O integrante precisa entender e saber explicar cada linha que for
> commitada.** Explique o que está fazendo, por que cada fórmula é assim e
> como a parte se encaixa no DQN. Prefira código simples e legível a código
> esperto. Se o integrante pedir só o código, entregue o código e ofereça uma
> explicação curta logo em seguida.

---

## 2. Regras obrigatórias

### 2.1 O contrato: `src/DQN/Types.hs`

- Todos os tipos compartilhados estão em `src/DQN/Types.hs`. **Use esses
  tipos.** Não crie tipos novos para representar algo que já existe lá (vetor,
  matriz, rede, camada, estado, ação, transição, buffer, agente,
  hiperparâmetros, log).
- **Não altere `Types.hs`.** Se faltar algo, o integrante abre uma *issue* ou
  um PR separado e o grupo decide. Mudar um tipo quebra o código de todo mundo.
- Cada módulo exporta **exatamente** as funções listadas na seção 4, com
  **exatamente** os nomes e as assinaturas de lá. Funções auxiliares internas
  são livres, mas não são exportadas.

### 2.2 Escopo: cada um mexe só na sua parte

Na branch de um integrante, a IA só pode alterar:

- o módulo da parte dele (`src/DQN/<Modulo>.hs`);
- o teste da parte dele (`test/DQN/<Modulo>Spec.hs`);
- o texto teórico da parte dele (`docs/teoria/NN-*.md`);
- na Parte 8, também `app/Main.hs`.

**Não altere** módulos de outras partes, `Types.hs`, o arquivo `.cabal`, a CI
nem este arquivo. Todos os módulos já estão listados no `.cabal`, então
ninguém precisa mexer nele.

Se o seu módulo depende de uma função de outra parte que ainda não foi
implementada, **não a implemente no seu módulo**. Use a assinatura da seção 4
e espere o PR da outra pessoa entrar (veja a ordem das ondas na seção 5).

### 2.3 Bibliotecas permitidas

Somente: `base`, `containers` (só `Data.Sequence`), `random`, `deepseq`. Nos
testes, também `hspec` e `QuickCheck`.

**Proibido** adicionar dependências: nada de `hmatrix`, `vector`, `hasktorch`,
`lens`, `mtl` etc. A rede neural é feita do zero, com listas. Isso é
proposital: (1) todo mundo compila sem instalar bibliotecas nativas e (2)
nada fica como caixa-preta na arguição.

### 2.4 Estilo de código

- **Identificadores em inglês** (seguindo `Types.hs`), **comentários e
  documentação em português**.
- **Toda função exportada** tem assinatura de tipo explícita e um comentário
  Haddock (`-- |`) em português dizendo o que faz. Em fórmulas, escreva a
  fórmula matemática no comentário.
- **Código puro.** Nada de `IO` fora de `app/Main.hs`. Aleatoriedade é sempre
  passada explicitamente: a função recebe um `StdGen` e devolve o `StdGen`
  novo junto com o resultado, no formato `(resultado, StdGen)`. Não use
  `IORef`, `unsafePerformIO`, `State`, `StateT` nem `randomIO`.
- **Funções totais quando possível.** Se uma função tem pré-condição (por
  exemplo, buffer não vazio), escreva isso no comentário.
- **Textos impressos no terminal sem acento** (mensagens do `Main`, nomes de
  testes do hspec). O terminal do Windows quebra o programa com caracteres
  como `ã` ou `ç`. Comentários no código podem ter acento normalmente.
- Sem `{-# LANGUAGE ... #-}` extra, exceto `BangPatterns` se precisar.
- Formatação: rode `ormolu --mode inplace <arquivo>` antes de commitar, se
  tiver o ormolu instalado. Indentação de 2 espaços.
- O código deve compilar **sem warnings** com `-Wall` (já está ligado no
  `.cabal`).

### 2.5 Testes

- Cada parte escreve os testes do próprio módulo em `test/DQN/<Modulo>Spec.hs`,
  usando `hspec`. O arquivo já existe com um teste `pending` para substituir.
- A seção 4 lista os testes mínimos de cada parte.
- `cabal test` precisa passar antes de abrir o PR. A CI roda o mesmo comando.

### 2.6 Commits e branches

- Uma branch por parte: `parte-N-nome`. Exemplo: `parte-1-env`.
- Nunca commitar direto na `main`. O merge é feito por Pull Request.
- Mensagens de commit em português, **sem acentos** (evita caracteres
  quebrados no terminal e no histórico).
- **Todo commit tem título E corpo.** O título é uma linha curta (até ~72
  caracteres) neste formato:
  - `feat(env): implementa a fisica do CartPole`
  - `test(env): adiciona testes de termino do episodio`
  - `docs(teoria): escreve secao de fundamentos de RL`
- Depois do título, uma linha em branco e um **corpo descrevendo o que foi
  feito**. O corpo responde:
  1. **O que foi feito:** quais funções ou seções foram implementadas ou alteradas.
  2. **Como:** as decisões de implementação mais importantes e o porquê
     (fórmula usada, estrutura de dados escolhida, cuidado tomado).
  3. **Como foi testado:** quais testes foram escritos e o resultado do
     `cabal test`.
- **Para a IA:** quando o integrante for commitar, escreva a mensagem
  completa (título + corpo) a partir do que foi realmente feito na sessão,
  mostre para ele revisar e só então faça o commit (ou entregue a mensagem
  para ele usar). Não invente no corpo algo que não está no diff.
- Para mensagens com corpo, use `git commit` (abre o editor) ou salve a
  mensagem num arquivo fora do repositório e use `git commit -F arquivo.txt`.
  Exemplo de mensagem completa:

  ```
  feat(replay): implementa o replay buffer com Data.Sequence

  Implementa emptyBuffer, pushTransition, bufferSize e sampleBatch,
  conforme a Parte 5 do AGENTS.md.

  - O buffer e uma fila FIFO: quando cheio, remove a transicao mais
    antiga (Seq.drop 1) antes de inserir a nova (|>).
  - sampleBatch sorteia indices uniformes em [0, tamanho - 1], com
    reposicao, passando o StdGen adiante a cada sorteio.
  - Usei Data.Sequence porque Seq.index e O(log n); com lista e (!!)
    o sorteio de 32 itens num buffer de 10.000 ficaria lento.

  Testes: capacidade maxima respeitada, FIFO com capacidade 3 e 5
  insercoes, sampleBatch devolve n itens presentes no buffer e o
  StdGen avanca. cabal test: 0 falhas.
  ```
- Cada integrante faz **no mínimo** um commit de código (`feat`/`test`) e um
  de teoria (`docs`), **da própria conta do GitHub**. A IA não faz commits em
  nome de outra pessoa.

---

## 3. Visão geral da arquitetura

```
             +-------------------------- DQN.Agent (Parte 7) --------------------------+
             |                                                                          |
  DQN.Env ---+--> estado s --> DQN.Policy (ε-greedy) --> ação a --> DQN.Env.stepEnv    |
  (Parte 1)  |                  (Parte 6)                              |                |
             |                      ^                                   v                |
             |                      |                    Transition (s, a, r, s', done)  |
             |               rede ONLINE                                |                |
             |            (DQN.Network, P3)                DQN.Replay (Parte 5)         |
             |                      ^                                   |                |
             |                      |                          minibatch sorteado       |
             |          DQN.Optimizer (Adam, P9)                        |                |
             |                      ^                                   v                |
             |          DQN.Backprop (P4)  <-- alvo y = r + γ·max Q_target(s', a')       |
             |                                        ^                                  |
             |                               rede TARGET (cópia a cada N passos)         |
             +--------------------------------------------------------------------------+
                                       |
                        DQN.Experiment + app/Main.hs (Parte 8): CSV, gráfico, avaliação

  DQN.Linalg (Parte 2): operações de vetor, matriz e rede usadas por quase todo mundo.
```

Rede usada: **4 entradas → 64 neurônios (ReLU) → 2 saídas (linear)**. Cada
saída é o Q-valor de uma ação: índice 0 = `PushLeft`, índice 1 = `PushRight`.

---

## 4. Especificação de cada parte

Para cada parte: o que o módulo faz, **as assinaturas exatas que ele exporta**,
o comportamento exigido, os erros comuns e os testes mínimos. A implementação
é livre, desde que cumpra isso.

---

### Parte 1: `DQN.Env` (o ambiente CartPole)

**Teoria:** `docs/teoria/01-fundamentos-rl.md` (agente, ambiente, estado, ação,
recompensa, episódio, política, função de valor, MDP).

**Responsabilidade:** simular o CartPole clássico (Barto, Sutton e Anderson,
1983), com a mesma física do `CartPole-v1` do Gymnasium.

```haskell
module DQN.Env (resetEnv, stepEnv, isTerminal, stateToVector) where

-- | Estado inicial: cada um dos 4 componentes sorteado uniformemente em [-0.05, 0.05].
resetEnv :: StdGen -> (CartState, StdGen)

-- | Aplica uma ação por um passo de tempo (0,02 s).
stepEnv :: CartState -> Action -> StepResult

-- | O episódio acabou? |posição| > 2.4  ou  |ângulo| > 12 graus (12·π/180 rad).
isTerminal :: CartState -> Bool

-- | Converte o estado no vetor de entrada da rede, NESTA ORDEM:
--   [posição, velocidade, ângulo, velocidade angular]
stateToVector :: CartState -> Vector
```

**Constantes:** gravidade g = 9.8; massa do carrinho m_c = 1.0; massa da haste
m_p = 0.1; massa total M = m_c + m_p; metade do comprimento da haste l = 0.5;
força F = +10.0 (`PushRight`) ou −10.0 (`PushLeft`); passo de tempo τ = 0.02.

**Equações** (θ = ângulo, θ̇ = velocidade angular, x = posição, ẋ = velocidade):

```
temp = (F + m_p · l · θ̇² · sin θ) / M
θ̈    = (g · sin θ − cos θ · temp) / ( l · (4/3 − m_p · cos² θ / M) )
ẍ    = temp − m_p · l · θ̈ · cos θ / M
```

Integração de Euler explícita (usa os valores **antigos** na direita):

```
x'  = x  + τ · ẋ        ẋ'  = ẋ  + τ · ẍ
θ'  = θ  + τ · θ̇        θ̇'  = θ̇  + τ · θ̈
```

**Recompensa:** 1.0 em **todo** passo, inclusive no passo em que o episódio
termina. `done = isTerminal s'`.

**Erros comuns:** misturar graus com radianos; atualizar x com a velocidade
nova em vez da antiga; inverter a ordem de `stateToVector`.

**Testes mínimos:**
- `resetEnv` devolve os 4 valores dentro de [-0.05, 0.05].
- Partindo do estado zero, `PushRight` deixa a velocidade do carrinho positiva
  e `PushLeft` deixa negativa.
- `isTerminal` é `True` para ângulo 0.25 rad e para posição 2.5, e `False` para
  o estado zero.
- Empurrando sempre para o mesmo lado, o episódio termina em menos de 100 passos.

---

### Parte 2: `DQN.Linalg` (álgebra linear e operações sobre redes)

**Teoria:** `docs/teoria/02-q-learning-ao-dqn.md` (Q-Learning tabular, equação
de Bellman, regra de atualização do Q-Learning e por que uma tabela não
funciona com estado contínuo como o do CartPole).

**Responsabilidade:** operações básicas que os outros módulos usam.
`Vector = [Double]` e `Matrix = [[Double]]` (lista de linhas).

```haskell
module DQN.Linalg
  ( dot, vAdd, vSub, vScale, hadamard
  , matVec, transposeM, outer, mAdd, mScale
  , argmax
  , zipNetworkWith, mapNetwork, addNetworks, scaleNetwork, zeroNetwork
  ) where

dot        :: Vector -> Vector -> Double      -- produto escalar: Σ aᵢ·bᵢ
vAdd       :: Vector -> Vector -> Vector      -- soma elemento a elemento
vSub       :: Vector -> Vector -> Vector      -- subtração elemento a elemento (a − b)
vScale     :: Double -> Vector -> Vector      -- k · v
hadamard   :: Vector -> Vector -> Vector      -- produto elemento a elemento
matVec     :: Matrix -> Vector -> Vector      -- matriz m×n vezes vetor de tamanho n → vetor de tamanho m
transposeM :: Matrix -> Matrix                -- transposta
outer      :: Vector -> Vector -> Matrix      -- produto externo: (outer a b) !! i !! j = aᵢ · bⱼ
mAdd       :: Matrix -> Matrix -> Matrix      -- soma elemento a elemento
mScale     :: Double -> Matrix -> Matrix      -- k · M
argmax     :: Vector -> Int                   -- índice do maior elemento; em EMPATE, o MENOR índice.
                                              -- Pré-condição: vetor não vazio.

-- Operações sobre redes inteiras (usadas para gradientes e pelo Adam).
-- As duas redes têm sempre o mesmo formato.
zipNetworkWith :: (Double -> Double -> Double) -> Network -> Network -> Network  -- aplica f peso a peso e bias a bias
mapNetwork     :: (Double -> Double) -> Network -> Network
addNetworks    :: Network -> Network -> Network   -- zipNetworkWith (+)
scaleNetwork   :: Double -> Network -> Network    -- multiplica todos os pesos e biases por k
zeroNetwork    :: Network -> Network              -- mesma forma, tudo zero
```

**Erros comuns:** em `outer`, trocar linhas por colunas (o resultado de
`outer a b` tem `length a` linhas); `argmax` devolvendo o maior índice no
empate.

**Testes mínimos:** `dot [1,2,3] [4,5,6] == 32`;
`matVec [[1,2],[3,4]] [1,1] == [3,7]`;
`outer [1,2] [3,4,5] == [[3,4,5],[6,8,10]]`; `transposeM` de uma matriz 2×3;
`argmax [1,3,3] == 1`; `zeroNetwork` mantém o formato e zera tudo;
`addNetworks n (zeroNetwork n) == n`.

---

### Parte 3: `DQN.Network` (a rede neural: inicialização e forward)

**Teoria:** `docs/teoria/03-redes-neurais.md` (neurônio, camada densa, ReLU,
forward pass, por que uma rede consegue aproximar Q(s, a), por que a última
camada é linear).

```haskell
module DQN.Network (relu, reluDerivative, initNetwork, forwardTrace, forward) where

relu           :: Double -> Double   -- max 0 z
reluDerivative :: Double -> Double   -- 1 se z > 0, senão 0 (inclusive em z = 0)

-- | Cria uma rede com os tamanhos de camada dados. Ex.: [4, 64, 2] cria 2 camadas
--   (4→64 e 64→2). Pesos: uniforme em [−limite, limite], com limite = sqrt (6 / n_entrada)
--   (inicialização de He). Biases: zero.
initNetwork :: [Int] -> StdGen -> (Network, StdGen)

-- | Forward pass guardando os valores intermediários: um LayerTrace por camada,
--   da primeira à última. Camadas ocultas: ReLU. ÚLTIMA camada: linear
--   (traceOutput == tracePreActivation).
forwardTrace :: Network -> Vector -> [LayerTrace]

-- | Saída da rede: os Q-valores, um por ação. É o traceOutput da última camada.
forward :: Network -> Vector -> Vector
```

Formato das camadas (ver `Types.hs`): para uma camada com n entradas e m
saídas, `weights` é m×n e `biases` tem m elementos. z = W·x + b.

**Erros comuns:** aplicar ReLU na última camada (aí a rede nunca dá Q-valor
negativo, e o aprendizado quebra); matriz de pesos n×m em vez de m×n; usar
o mesmo gerador para todos os pesos (tudo igual).

**Testes mínimos:** `initNetwork [4,64,2]` gera 2 camadas com pesos 64×4 e
2×64 e biases de tamanho 64 e 2; os pesos estão dentro do limite e os biases
são zero; `forward` devolve vetor de tamanho 2; uma rede montada à mão com
pesos conhecidos dá a saída calculada no papel; a última camada consegue dar
saída negativa; `length (forwardTrace n x) == length (layers n)`.

---

### Parte 4: `DQN.Backprop` (perda e gradientes)

**Teoria:** `docs/teoria/04-perda-e-backprop.md` (alvo de Bellman, erro TD,
função de perda de Huber vs. erro quadrático, gradiente descendente, regra da
cadeia, backpropagation).

```haskell
module DQN.Backprop (huberLoss, huberLossDerivative, backprop, sampleGradient, batchGradient) where

-- | Perda de Huber com δ = 1, recebendo (previsão, alvo). Com d = previsão − alvo:
--   0.5·d²  se |d| ≤ 1;   |d| − 0.5  caso contrário.
huberLoss :: Double -> Double -> Double

-- | Derivada da Huber em relação à previsão: d limitado ao intervalo [−1, 1].
huberLossDerivative :: Double -> Double -> Double

-- | Backpropagation. Recebe a rede, a entrada x e o gradiente da perda em
--   relação à SAÍDA da rede (dL/dsaída). Devolve o gradiente da perda em
--   relação a TODOS os pesos e biases, no formato de uma Network.
backprop :: Network -> Vector -> Vector -> Network

-- | Gradiente e perda para UMA amostra: entrada x, índice da ação a, alvo y.
--   A perda só olha a saída da ação a: L = huberLoss (Q(x)[a]) y.
--   As outras saídas recebem gradiente 0.
sampleGradient :: Network -> Vector -> Int -> Double -> (Network, Double)

-- | Média dos gradientes e média das perdas de um minibatch de (x, a, y).
--   Pré-condição: lista não vazia.
batchGradient :: Network -> [(Vector, Int, Double)] -> (Network, Double)
```

**Backpropagation, camada por camada, da última para a primeira**
(dA = gradiente em relação à saída da camada, começando com dL/dsaída):

```
dz      = dA                                   (última camada, linear)
dz      = dA ⊙ reluDerivative(z)               (camadas ocultas)
dW      = outer dz entrada_da_camada           (mesmo formato de W)
db      = dz
dA_ant  = Wᵀ · dz                              (vira o dA da camada anterior)
```

Use `forwardTrace` (Parte 3) para ter z e as entradas de cada camada, e as
funções da Parte 2.

**Erros comuns:** esquecer de multiplicar pela derivada da ReLU; usar a saída
depois da ReLU em vez de z na derivada; devolver as camadas na ordem
invertida; calcular a perda em todas as saídas em vez de só na da ação
escolhida.

**Testes mínimos:**
- **Verificação numérica do gradiente (obrigatório).** Para uma rede pequena
  ([4,5,3,2] por exemplo), compare alguns elementos de `sampleGradient` com
  (L(w + h) − L(w)) / h, com h = 1e-6. Devem bater com tolerância ~1e-4.
  Esse teste prova que o backprop está certo.
- Gradiente zero quando a previsão já é igual ao alvo.
- `huberLoss` e `huberLossDerivative` em valores dentro e fora de [−1, 1].
- `batchGradient` de uma lista com uma amostra é igual a `sampleGradient`.

---

### Parte 5: `DQN.Replay` (experience replay)

**Teoria:** `docs/teoria/05-experience-replay.md` (por que treinar com
experiências em sequência é ruim, correlação entre amostras, reaproveitamento
de dados, buffer circular).

```haskell
module DQN.Replay (emptyBuffer, pushTransition, bufferSize, sampleBatch) where

-- | Buffer vazio com a capacidade dada.
emptyBuffer :: Int -> ReplayBuffer

-- | Adiciona uma transição. Se o buffer está cheio, remove a MAIS ANTIGA (FIFO).
pushTransition :: Transition -> ReplayBuffer -> ReplayBuffer

-- | Quantas transições há no buffer.
bufferSize :: ReplayBuffer -> Int

-- | Sorteia n transições de forma uniforme, COM reposição (a mesma pode sair
--   duas vezes). Pré-condição: buffer não vazio.
sampleBatch :: Int -> ReplayBuffer -> StdGen -> ([Transition], StdGen)
```

Use as funções de `Data.Sequence` (`|>`, `Seq.drop`, `Seq.index`,
`Seq.length`). Sortear de uma lista com `!!` é lento demais com 10.000 itens.

**Erros comuns:** remover a transição mais nova em vez da mais antiga;
sortear índices até `bufferSize` (inclusive), o que sai do intervalo; reusar
o mesmo `StdGen` em todos os sorteios (sai sempre a mesma transição).

**Testes mínimos:** o tamanho nunca passa da capacidade; com capacidade 3 e
5 inserções, sobram as 3 últimas; `sampleBatch 32` devolve 32 transições,
todas presentes no buffer; o `StdGen` devolvido é diferente do recebido.

---

### Parte 6: `DQN.Policy` (exploração: ε-greedy)

**Teoria:** `docs/teoria/06-exploracao.md` (exploration vs. exploitation,
política gulosa, ε-greedy, decaimento do ε e por que o ε não vai a zero).

```haskell
module DQN.Policy (epsilonAt, greedyAction, epsilonGreedy) where

-- | ε depois de t passos de treino: decai LINEARMENTE de hpEpsilonStart até
--   hpEpsilonEnd em hpEpsilonDecaySteps passos, e depois fica constante em hpEpsilonEnd.
epsilonAt :: Hyperparams -> Int -> Double

-- | Ação com maior Q-valor segundo a rede: indexToAction (argmax (forward rede s)).
greedyAction :: Network -> Vector -> Action

-- | Com probabilidade ε, ação aleatória (uniforme entre as ações);
--   senão, ação gulosa. Sorteia u uniforme em [0, 1): se u < ε, explora.
epsilonGreedy :: Double -> Network -> Vector -> StdGen -> (Action, StdGen)
```

**Erros comuns:** ε decaindo por episódio em vez de por passo; ε indo a zero;
na exploração, sortear a ação com o mesmo número u usado para decidir se
explora.

**Testes mínimos:** `epsilonAt hp 0 == hpEpsilonStart`;
`epsilonAt hp hpEpsilonDecaySteps == hpEpsilonEnd`; depois disso continua igual
a `hpEpsilonEnd`; no meio do caminho fica no valor intermediário; com ε = 0,
`epsilonGreedy` sempre devolve `greedyAction`; com ε = 1, em 200 sorteios as
duas ações aparecem.

---

### Parte 7: `DQN.Agent` (o algoritmo DQN completo)

**Teoria:** `docs/teoria/07-arquitetura-dqn.md` (o algoritmo do artigo da
DeepMind, Mnih et al. 2015, passo a passo; rede online vs. target network e
por que a target existe; o loop de treinamento).

```haskell
module DQN.Agent (newAgent, tdTarget, trainStep, runEpisode, train) where

-- | Agente novo: rede online criada com initNetwork (hpLayerSizes hp);
--   target = cópia da online; buffer vazio com capacidade hpBufferCapacity;
--   otimizador = initAdam da online; 0 passos; agentBest = online; agentBestScore = 0.
newAgent :: Hyperparams -> StdGen -> (Agent, StdGen)

-- | Alvo de Bellman de UMA transição, calculado com a rede TARGET:
--   se tDone:   y = r
--   senão:      y = r + γ · max_a' Q_target(s', a')
tdTarget :: Double -> Network -> Transition -> Double

-- | Um passo de treino com um minibatch: calcula os alvos com a target network,
--   o gradiente da rede online com batchGradient e atualiza a online com adamStep.
--   Devolve o agente atualizado e a perda média. NÃO mexe em target, buffer nem passos.
trainStep :: Hyperparams -> Agent -> [Transition] -> (Agent, Double)

-- | Roda um episódio completo (o Int é o número do episódio, para o log).
runEpisode :: Hyperparams -> Int -> Agent -> StdGen -> (Agent, EpisodeLog, StdGen)

-- | Treino completo: newAgent e depois hpEpisodes episódios, numerados de 1 em diante.
--   Devolve o agente final e um log por episódio, em ordem.
train :: Hyperparams -> StdGen -> (Agent, [EpisodeLog])
```

**`runEpisode`, a cada passo** (até `done` ou `hpMaxStepsPerEpisode` passos):

1. ε = `epsilonAt hp agentTotalSteps`; escolhe a ação com `epsilonGreedy`
   usando a rede **online**.
2. `stepEnv`; monta a `Transition` (com `stateToVector`) e coloca no buffer.
3. `agentTotalSteps` + 1.
4. Se `bufferSize ≥ hpWarmupSteps`: sorteia `hpBatchSize` transições e chama
   `trainStep`.
5. Se `agentTotalSteps` é múltiplo de `hpTargetUpdateEvery`: target := online.

O `EpisodeLog` tem: soma das recompensas, ε no fim do episódio e média das
perdas dos passos de treino (0 se não treinou nenhuma vez).

**`train`, depois de cada episódio:** calcula a média das recompensas dos
últimos `hpBestWindow` episódios. Se já há `hpBestWindow` episódios e essa
média é maior que `agentBestScore`, então `agentBest := agentOnline` e
`agentBestScore := média`.

**IMPORTANTE (desempenho):** Haskell é preguiçoso. Sem cuidado, o programa
acumula milhões de cálculos pendentes e estoura a memória. A cada passo,
force a avaliação do agente com `force` de `Control.DeepSeq` (os tipos já
derivam `NFData`) e use `BangPatterns` nos acumuladores do loop.

**Erros comuns:** calcular o alvo com a rede online em vez da target;
bootstrapping (somar γ·max Q) em transição terminal; não sincronizar a target
nunca; esquecer o `force`.

**Testes mínimos:** `tdTarget` com `tDone = True` devolve só a recompensa;
com `False`, devolve r + γ·max calculado à mão; repetir `trainStep` no mesmo
minibatch faz a perda cair; `trainStep` não altera `agentTarget`; `train` com
hiperparâmetros pequenos (5 episódios, warmup 10, batch 4) devolve 5 logs
numerados de 1 a 5.

---

### Parte 8: `DQN.Experiment` + `app/Main.hs` (experimento e resultados)

**Teoria:** `docs/teoria/08-experimento-e-resultados.md` (como o experimento
foi montado, hiperparâmetros e por quê, a curva de aprendizado, a comparação
com o agente aleatório e a análise dos resultados).

```haskell
module DQN.Experiment (logsToCsv, movingAverage, evaluatePolicy, asciiChart) where

-- | CSV com cabeçalho exatamente "episode,total_reward,epsilon,avg_loss",
--   uma linha por episódio.
logsToCsv :: [EpisodeLog] -> String

-- | Média móvel: o elemento i é a média dos até w elementos que terminam em i
--   (no começo, usa os que existem). Mesmo tamanho da lista de entrada.
movingAverage :: Int -> [Double] -> [Double]

-- | Roda n episódios só com a política GULOSA (ε = 0, sem treinar) e devolve
--   a recompensa de cada um. Respeita hpMaxStepsPerEpisode.
evaluatePolicy :: Hyperparams -> Network -> Int -> StdGen -> ([Double], StdGen)

-- | Gráfico de barras em texto (largura × altura), para o terminal.
asciiChart :: Int -> Int -> [Double] -> String
```

**`app/Main.hs`** (é o único lugar com `IO`):

- Argumentos opcionais: `semente` (padrão 42) e `episodios` (padrão
  `hpEpisodes`). Exemplo: `cabal run dqn -- 1 400`.
- Treina, salva `results/training.csv` (cria a pasta se não existir), mostra
  no terminal o gráfico da média móvel de 20 episódios e avalia, com 20
  episódios cada, a **política aleatória**, a **rede final** e a **melhor rede**
  (`agentBest`), mostrando a média de cada uma.

**Testes mínimos:** cabeçalho do CSV e número de linhas; `movingAverage 2
[2,4,6] == [2,3,5]`; `evaluatePolicy` devolve n valores, todos entre 1 e
`hpMaxStepsPerEpisode`; `asciiChart` tem `altura` linhas.

---

### Parte 9: `DQN.Optimizer` (SGD e Adam)

**Teoria:** `docs/teoria/09-otimizacao-limitacoes-aplicacoes.md` (gradiente
descendente vs. Adam; vantagens e limitações do DQN: instabilidade,
superestimação dos Q-valores, custo de amostras, ações discretas; variantes:
Double DQN, Dueling DQN, Prioritized Replay, Rainbow; aplicações: Atari,
robótica, recomendação etc.).

```haskell
module DQN.Optimizer (initAdam, sgdStep, adamStep) where

-- | Estado inicial do Adam para uma rede: momentos zerados (zeroNetwork) e contador 0.
initAdam :: Network -> AdamState

-- | Gradiente descendente simples: w ← w − lr · g, para cada peso e bias.
--   (Taxa de aprendizado, rede, gradiente.)
sgdStep :: Double -> Network -> Network -> Network

-- | Um passo do Adam. Recebe taxa de aprendizado, estado, rede e gradiente;
--   devolve a rede nova e o estado novo.
adamStep :: Double -> AdamState -> Network -> Network -> (Network, AdamState)
```

**Adam** (Kingma e Ba, 2014), com β₁ = 0.9, β₂ = 0.999, ε = 1e-8, aplicado a
cada peso w com gradiente g:

```
t' = t + 1
m' = β₁·m + (1 − β₁)·g
v' = β₂·v + (1 − β₂)·g²
m̂  = m' / (1 − β₁^t')
v̂  = v' / (1 − β₂^t')
w' = w − lr · m̂ / (√v̂ + ε)
```

Use `zipNetworkWith` da Parte 2 para aplicar as fórmulas em todos os pesos.

**Erros comuns:** esquecer a correção de viés (m̂, v̂); usar t em vez de t' na
correção (no primeiro passo dá divisão por zero); confundir o ε do Adam com o
ε da exploração.

**Testes mínimos:** `initAdam` tem momentos zerados e contador 0; o contador
sobe 1 a cada passo; no primeiro passo, cada peso anda ≈ lr na direção oposta
ao sinal do gradiente; `sgdStep` com gradiente zero não muda a rede; aplicar
`adamStep` repetidamente em f(w) = w² (gradiente 2w) leva w para perto de 0.

---

## 5. Ordem de integração (ondas)

Todos podem programar ao mesmo tempo, mas um PR só passa na CI depois que as
partes das quais ele depende já entraram na `main`:

| Onda | Partes | Dependem de |
|---|---|---|
| 1 | 1 Env, 2 Linalg, 5 Replay | só `Types` |
| 2 | 3 Network, 9 Optimizer | Linalg |
| 3 | 4 Backprop, 6 Policy | Network |
| 4 | 7 Agent | todas acima |
| 5 | 8 Experiment + Main | Agent |

Os commits de teoria (`docs/teoria/`) não dependem de nada e podem entrar a
qualquer momento.

---

## 6. Texto teórico de cada parte

Cada parte escreve um arquivo em `docs/teoria/`, seguindo o modelo que já está
lá. Esse texto é a base dos slides da parte e do estudo para a arguição.

- Escrito em português, com as próprias palavras, não copiado da internet.
- **Liga a teoria ao nosso código:** cite o módulo e as funções que
  implementam cada conceito. A avaliação cobra a relação entre teoria e prática.
- Inclui fórmulas quando houver e um exemplo numérico pequeno.
- Termina com **perguntas prováveis da arguição e as respostas**.

---

## 7. Resultados esperados (referência)

Com `defaultHyperparams`, compilando com otimização (`-O2`, já configurado):

- Política aleatória: ~20 a 25 passos por episódio.
- O treino de 400 episódios leva de 1 a 3 minutos num notebook comum.
- O agente costuma aprender entre os episódios 200 e 350, com média móvel
  passando de 150 passos. Às vezes ele piora depois. Isso é a instabilidade
  conhecida do DQN, não necessariamente um bug, e é por isso que guardamos
  a `agentBest`.
- A melhor rede, avaliada sem exploração, deve fazer bem mais que a aleatória
  (centenas de passos na maioria das sementes).

Se o seu resultado ficar sempre perto de 10 passos ou 20 passos depois de
400 episódios, provavelmente há um bug. Os suspeitos mais comuns: o teste
numérico de gradiente da Parte 4, a ordem de `stateToVector`, ReLU na última
camada e o alvo calculado com a rede online.
