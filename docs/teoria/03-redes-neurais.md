# Redes neurais como aproximadoras da função Q

> **Parte 3** · Responsável: Felipe Massola Fonseca Leite · Código relacionado: `DQN.Network`

## O que esta seção precisa cobrir

- [x] Neurônio artificial e camada densa: z = W·x + b
- [x] Função de ativação ReLU e por que precisamos de não linearidade
- [x] Forward pass, com um exemplo numérico pequeno
- [x] Nossa arquitetura: 4 → 64 (ReLU) → 2 (linear). Por que a saída é linear e por que há uma saída por ação
- [x] Inicialização dos pesos (He) e por que não iniciar tudo com zero

## Conceito

No Q-Learning tabular, guardamos um valor $Q(s, a)$ para cada par estado-ação em uma tabela. No CartPole, porém, o estado é um vetor contínuo de 4 números reais $(x, \dot{x}, \theta, \dot{\theta})$, o que torna impossível construir uma tabela. A solução do **Deep Q-Learning (DQN)** é usar uma **rede neural artificial** como aproximadora de função: ela recebe o estado $s$ e devolve uma estimativa dos Q-valores para cada ação possível.

A unidade básica da rede é o **neurônio artificial**. Ele recebe um vetor de entradas $x$, calcula uma combinação linear usando um vetor de pesos $w$ e um viés (*bias*) $b$, e opcionalmente passa o resultado por uma **função de ativação** não linear. Quando agrupamos $m$ neurônios em paralelo recebendo as mesmas $n$ entradas, formamos uma **camada densa** (ou totalmente conectada), cuja operação é escrita de forma compacta como multiplicação de matriz por vetor somada ao bias: $z = W \cdot x + b$.

Para que uma rede com várias camadas seja mais expressiva do que uma única transformação linear, precisamos de uma **função de ativação não linear** nas camadas ocultas. Sem ela, empilhar duas camadas lineares $W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)$ equivale matematicamente a uma única camada linear, incapaz de modelar superfícies de decisão complexas como a do CartPole. Usamos a **ReLU** (*Rectified Linear Unit*), que zera valores negativos e mantém positivos. Ela é simples, barata de calcular e evita o desaparecimento do gradiente para ativações positivas.

O processo de passar uma entrada $x$ da primeira até a última camada para obter a saída da rede é chamado de **forward pass** (propagação para frente).

No nosso projeto, usamos a arquitetura **4 → 64 (ReLU) → 2 (linear)**:

- **4 entradas:** correspondem exatamente ao vetor de estado do CartPole $[x, \dot{x}, \theta, \dot{\theta}]$.
- **64 neurônios ocultos com ReLU:** camada intermediária com capacidade suficiente para capturar a dinâmica não linear do equilíbrio da haste, mantendo o treino rápido em Haskell puro.
- **2 saídas lineares (uma por ação):** índice `0` = $Q(s, \text{PushLeft})$ e índice `1` = $Q(s, \text{PushRight})$. Ter uma saída por ação (em vez de passar a ação como entrada) permite calcular os Q-valores de **todas** as ações em um único *forward pass*, o que torna operações como $\arg\max_a Q(s, a)$ e $\max_{a'} Q(s', a')$ muito mais eficientes.
- **Por que a última camada é linear (sem ReLU)?** Porque Q-valores representam retornos esperados e podem assumir **qualquer valor real**, inclusive negativo (em ambientes com recompensas negativas ou durante ajustes intermediários do treino). Se aplicássemos ReLU na saída, a rede jamais conseguiria produzir valores negativos e teria gradiente zero sempre que a pré-ativação fosse menor ou igual a zero, travando o aprendizado.

Por fim, antes de começar a treinar, precisamos definir os valores iniciais dos pesos. **Não podemos iniciar todos os pesos com zero**, porque isso cria **simetria perfeita**: todos os neurônios da camada oculta receberiam a mesma entrada, produziriam a mesma saída e teriam exatamente o mesmo gradiente no backpropagation, permanecendo idênticos para sempre (como se houvesse apenas 1 neurônio). Usamos a **inicialização de He** (uniforme), projetada especificamente para camadas com ReLU: ela sorteia pesos em torno de zero com amplitude proporcional a $1/\sqrt{n_{\text{entrada}}}$, mantendo a variância das ativações estável entre as camadas. Os biases podem começar em zero sem quebrar a simetria, já que os pesos já são diferentes.

## Fórmulas

**Camada densa** (com $n$ entradas e $m$ saídas, onde $W \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^n$ e $b \in \mathbb{R}^m$):

$$z = W \cdot x + b, \qquad z_i = \sum_{j=1}^{n} W_{ij} x_j + b_i$$

**Função de ativação ReLU e sua derivada:**

$$\text{ReLU}(z) = \max(0, z)$$

$$\text{ReLU}'(z) = \begin{cases} 1, & \text{se } z > 0 \\ 0, & \text{se } z \le 0 \end{cases}$$

**Forward pass da nossa rede (2 camadas):**

$$z^{(1)} = W^{(1)} x + b^{(1)}, \qquad a^{(1)} = \text{ReLU}(z^{(1)})$$

$$z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}, \qquad \hat{Q}(s, \cdot) = z^{(2)}$$

**Inicialização de He (distribuição uniforme):** para uma camada com $n_{\text{in}}$ entradas e ativação ReLU, queremos $\text{Var}(W) = \frac{2}{n_{\text{in}}}$. Como a variância de uma distribuição uniforme em $[-L, L]$ é $\frac{L^2}{3}$, igualar $\frac{L^2}{3} = \frac{2}{n_{\text{in}}}$ resulta no limite $L$:

$$L = \sqrt{\frac{6}{n_{\text{in}}}}, \qquad W_{ij} \sim \mathcal{U}(-L, L), \qquad b_i = 0$$

## Exemplo

Considere uma rede pequena com arquitetura **2 → 2 (ReLU) → 2 (linear)** e entrada $x = [2.0,\; 1.0]$:

- **Camada 1 (oculta, com ReLU):**
  $$W^{(1)} = \begin{bmatrix} 1 & -2 \\ -1 & 3 \end{bmatrix}, \qquad b^{(1)} = \begin{bmatrix} 0.5 \\ -1.0 \end{bmatrix}$$

  Calculamos a pré-ativação $z^{(1)} = W^{(1)} x + b^{(1)}$:
  - $z^{(1)}_1 = 1 \cdot 2.0 + (-2) \cdot 1.0 + 0.5 = 0.5$
  - $z^{(1)}_2 = (-1) \cdot 2.0 + 3 \cdot 1.0 + (-1.0) = 0.0$

  Aplicando ReLU elemento a elemento:
  - $a^{(1)} = [\max(0, 0.5),\; \max(0, 0.0)] = [0.5,\; 0.0]$

- **Camada 2 (saída, linear):**
  $$W^{(2)} = \begin{bmatrix} 2 & 4 \\ -3 & 1 \end{bmatrix}, \qquad b^{(2)} = \begin{bmatrix} 1.0 \\ -0.5 \end{bmatrix}$$

  Calculamos a saída $z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}$:
  - $z^{(2)}_1 = 2 \cdot 0.5 + 4 \cdot 0.0 + 1.0 = 2.0$
  - $z^{(2)}_2 = (-3) \cdot 0.5 + 1 \cdot 0.0 + (-0.5) = -2.0$

Como a última camada é linear, a saída final da rede é $[2.0,\; -2.0]$. Note que o segundo Q-valor é negativo, o que só é possível porque não aplicamos ReLU na última camada. A política gulosa escolheria a ação de índice `0` ($2.0 > -2.0$).

## Como isso aparece no nosso código

Os tipos da rede ficam em `src/DQN/Types.hs` (`Layer`, `Network` e `LayerTrace`) e a lógica de ativação, inicialização e *forward pass* está implementada em `src/DQN/Network.hs`, apoiada nas operações de álgebra linear de `DQN.Linalg`:

- **`relu` e `reluDerivative`:** implementam $\max(0, z)$ e sua derivada por partes (`1.0` para `z > 0` e `0.0` para `z <= 0`). `relu` é usada no *forward pass* das camadas ocultas e `reluDerivative` é exportada para o backpropagation (`DQN.Backprop`).
- **`initNetwork :: [Int] -> StdGen -> (Network, StdGen)`:** recebe a lista de tamanhos das camadas (por exemplo, `[4, 64, 2]`) e constrói os pares `(nIn, nOut)` adjacentes. Para cada camada, calcula `limit = sqrt (6.0 / fromIntegral nIn)`, gera uma matriz `nOut × nIn` com `randomR (-limit, limit)` encadeando o `StdGen` a cada peso (código 100% puro, sem repetir o mesmo gerador) e inicializa `biases` com zeros.
- **`forwardTrace :: Network -> Vector -> [LayerTrace]`:** percorre as camadas calculando `z = vAdd (matVec (weights layer) x) (biases layer)`. Quando há camadas seguintes (`layer : rest`), aplica `map relu z`; quando chega na última camada (`[lastLayer]`), mantém `traceOutput = z` (ativação linear). Guardar `traceInput`, `tracePreActivation` e `traceOutput` de cada camada evita recomputar o *forward pass* no backpropagation (`DQN.Backprop`).
- **`forward :: Network -> Vector -> Vector`:** reutiliza `forwardTrace` e devolve o `traceOutput` da última camada — o vetor de Q-valores usado por `DQN.Policy` (para escolher ações) e `DQN.Agent` (para calcular alvos de Bellman).

## Perguntas prováveis da arguição

1. **Pergunta:** Por que não aplicamos ReLU na última camada da rede?
   **Resposta:** Porque a última camada estima os Q-valores $Q(s, a)$, que podem ser quaisquer números reais, positivos ou negativos. A ReLU cortaria qualquer saída negativa para zero, impedindo a rede de representar Q-valores negativos e zerando o gradiente nessas regiões.

2. **Pergunta:** O que aconteceria se inicializássemos todos os pesos da rede com zero?
   **Resposta:** Todos os neurônios da camada oculta receberiam as mesmas entradas com pesos iguais, gerando exatamente a mesma saída e recebendo o mesmo gradiente durante o backpropagation. Essa simetria nunca seria quebrada, e os 64 neurônios se comportariam como se fossem apenas 1.

3. **Pergunta:** Por que usamos a inicialização de He ($\sqrt{6 / n_{\text{entrada}}}$) em vez de sortear pesos em um intervalo fixo qualquer, como $[-10, 10]$?
   **Resposta:** Se os pesos iniciais forem muito grandes, as ativações e gradientes podem explodir logo no início; se forem muito pequenos, o sinal enfraquece ao atravessar as camadas. Como a ReLU zera aproximadamente metade das entradas (as negativas), a inicialização de He escala o intervalo de acordo com o número de entradas da camada para preservar a variância das ativações.

4. **Pergunta:** Por que o módulo tem `forwardTrace` além de `forward`?
   **Resposta:** Durante a escolha de ação (`DQN.Policy`), só precisamos da saída final (`forward`). Porém, durante o treinamento (`DQN.Backprop`), a regra da cadeia precisa da entrada de cada camada (`traceInput`) e da pré-ativação $z$ (`tracePreActivation`) para calcular a derivada da ReLU e o produto externo $dW = dz \otimes x$. O `forwardTrace` guarda exatamente esses valores intermediários.

5. **Pergunta:** Por que a matriz de pesos de uma camada com $n$ entradas e $m$ saídas tem dimensão $m \times n$ e não $n \times m$?
   **Resposta:** Porque cada linha da matriz corresponde aos pesos de um neurônio de saída. Assim, o produto matriz-vetor `matVec W x` faz o produto escalar de cada uma das $m$ linhas (de tamanho $n$) com o vetor de entrada $x$ (de tamanho $n$), produzindo diretamente um vetor de tamanho $m$ para somar com os $m$ biases.

## Referências

- He, K.; Zhang, X.; Ren, S.; Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. ICCV.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Sutton, R. S.; Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. 2ª ed. MIT Press.
