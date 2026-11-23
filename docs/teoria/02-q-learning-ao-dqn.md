# Do Q-Learning tabular ao Deep Q-Learning

> **Parte 2** · Responsável: Bruno Santana de Negreiros · Código relacionado: `DQN.Linalg`

## O que esta seção precisa cobrir

- [x] Q-Learning tabular: a tabela Q(s, a)
- [x] Equação de Bellman e regra de atualização do Q-Learning (com fórmula)
- [x] Por que a tabela não funciona quando o estado é contínuo (o estado do CartPole tem 4 números reais)
- [x] A ideia do DQN: trocar a tabela por uma função que aproxima Q
- [x] Papel da álgebra linear (vetores, matrizes, produto externo) dentro da rede

## Conceito

No aprendizado por reforço, o agente precisa decidir qual ação tomar em cada estado para acumular a maior recompensa possível ao longo do tempo. O **Q-Learning tabular** resolve isso mantendo uma tabela chamada $Q(s, a)$: cada linha representa um estado $s$, cada coluna representa uma ação $a$, e a célula guarda uma estimativa da recompensa total futura esperada caso o agente tome a ação $a$ no estado $s$ e siga agindo de forma ótima depois disso.

Para aprender os valores dessa tabela sem conhecer as regras do ambiente de antemão, o agente interage com o ambiente passo a passo. A cada transição $(s, a, r, s')$, ele compara o valor que achava que a ação tinha ($Q(s, a)$) com uma estimativa mais atualizada baseada na recompensa real recebida ($r$) somada ao melhor retorno estimado a partir do próximo estado ($s'$). A diferença entre esses dois valores (o *erro de diferença temporal*, ou erro TD) corrige gradualmente a célula $Q(s, a)$ na tabela.

O problema aparece quando saímos de ambientes pequenos e discretos (como um tabuleiro) para ambientes com **estados contínuos**, como o **CartPole**. No CartPole, cada estado é composto por quatro números reais: posição do carrinho, velocidade do carrinho, ângulo da haste e velocidade angular da haste. Como existem infinitas combinações de quatro números reais, seria impossível criar uma linha de tabela para cada estado; mesmo discretizando em faixas, o número de células explode exponencialmente (*maldição da dimensionalidade*) e o agente não consegue generalizar o que aprendeu de um estado para outro quase idêntico.

O **Deep Q-Learning (DQN)** resolve essa limitação substituindo a tabela por uma **rede neural parametrizada** $Q(s, a; \theta)$, onde $\theta$ são os pesos e biases da rede. Em vez de armazenar um número separado para cada estado, a rede recebe o vetor contínuo de 4 entradas do estado $s$ e calcula, via operações de álgebra linear, um vetor de saída com um Q-valor estimado para cada ação possível (`PushLeft` e `PushRight`). Assim, estados parecidos ativam neurônios parecidos e geram estimativas coerentes mesmo para estados nunca vistos antes.

Dentro dessa rede, a **álgebra linear** é o motor de todo o processo: vetores representam estados, ativações e biases; matrizes guardam os pesos das conexões entre camadas; o produto matriz-vetor calcula a passagem direta (*forward pass*) de cada camada; a transposição de matrizes e o produto de Hadamard propagam o erro para trás; e o **produto externo** combina o erro de cada neurônio com a entrada da camada para produzir a matriz de gradientes que ajusta os pesos.

## Fórmulas

### Equação de otimalidade de Bellman para $Q^*$

$$Q^*(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^*(s', a') \;\middle|\; s, a \right]$$

### Regra de atualização do Q-Learning tabular

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left( \underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{alvo TD}} - Q(s, a) \right)$$

### Aproximação de função no DQN

$$Q(s, a; \theta) \approx Q^*(s, a)$$

Onde:
- $s$: estado atual;
- $a$: ação escolhida no estado $s$;
- $r$: recompensa imediata recebida após executar $a$ em $s$;
- $s'$: estado seguinte retornado pelo ambiente;
- $a'$: ações possíveis no estado $s'$;
- $\gamma \in [0, 1)$: fator de desconto, que controla o quanto recompensas futuras valem em relação à recompensa imediata (no nosso projeto, $\gamma = 0{,}99$);
- $\alpha \in (0, 1]$: taxa de aprendizado, que controla o tamanho do passo de atualização na tabela;
- $\theta$: conjunto de pesos $W$ e biases $b$ da rede neural.

### Operações de álgebra linear na rede

- **Camada densa (forward pass):** $z = W \cdot x + b$, com $z_i = \sum_j W_{ij} x_j + b_i$.
- **Gradiente dos pesos (produto externo):** $dW = dz \otimes x$, onde $(dW)_{ij} = dz_i \cdot x_j$.
- **Propagação do erro para a camada anterior:** $dA_{\text{ant}} = W^T \cdot dz$.
- **Modulação pela derivada da ativação (Hadamard):** $dz = dA \odot \text{ReLU}'(z)$.

## Exemplo

Suponha um ambiente simples com taxa de aprendizado $\alpha = 0{,}5$ e fator de desconto $\gamma = 0{,}9$.

1. Na tabela atual, no estado $s_0$, temos $Q(s_0, \text{Esq}) = 1{,}0$ e $Q(s_0, \text{Dir}) = 0{,}5$.
2. O agente executa a ação $\text{Esq}$, recebe recompensa $r = 1{,}0$ e vai para o estado $s_1$.
3. No estado $s_1$, os valores atuais da tabela são $Q(s_1, \text{Esq}) = 2{,}0$ e $Q(s_1, \text{Dir}) = 3{,}0$. Logo, $\max_{a'} Q(s_1, a') = 3{,}0$.
4. Calculamos o alvo TD:
   $$y = r + \gamma \max_{a'} Q(s_1, a') = 1{,}0 + 0{,}9 \times 3{,}0 = 3{,}7$$
5. Atualizamos a célula $Q(s_0, \text{Esq})$:
   $$Q(s_0, \text{Esq}) \leftarrow 1{,}0 + 0{,}5 \times (3{,}7 - 1{,}0) = 1{,}0 + 1{,}35 = 2{,}35$$

No DQN, em vez de atualizar uma célula isolada da tabela, a rede calcula os Q-valores via produto matriz-vetor. Por exemplo, com uma camada linear pequena de pesos $W = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, bias $b = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$ e entrada $x = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$:

$$z = W \cdot x + b = \begin{bmatrix} 1\cdot 1 + 2\cdot 1 \\ 3\cdot 1 + 4\cdot 1 \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix} = \begin{bmatrix} 3 \\ 6 \end{bmatrix}$$

A ação escolhida pela política gulosa é $\arg\max([3, 6]) = 1$ (`PushRight`). Para corrigir os pesos depois de obter o vetor de erro $dz$, o produto externo entre $dz$ e a entrada $x$ fornece diretamente a matriz de ajuste para cada peso de $W$.

## Como isso aparece no nosso código

No projeto, o estado do CartPole (`CartState` em `DQN.Types`) é convertido em um `Vector` de 4 `Double`s, e a rede (`Network`) é composta por camadas (`Layer`) com matriz de pesos `Matrix` ($m \times n$) e vetor de `biases` (`Vector`). Toda a base matemática usada pelos módulos seguintes está implementada em `DQN.Linalg`:

- **`dot`, `matVec` e `vAdd`**: usados em `DQN.Network` no cálculo do *forward pass* de cada camada ($z = W \cdot x + b$), transformando o vetor de estado contínuo de 4 elementos nos 2 Q-valores de saída.
- **`argmax`**: usado em `DQN.Policy` (`greedyAction`) para escolher o índice da ação de maior Q-valor na saída da rede (resolvendo empates pelo menor índice).
- **`outer`, `transposeM` e `hadamard`**: usados em `DQN.Backprop` durante o backpropagation — `hadamard` aplica a derivada da ReLU elemento a elemento, `outer` gera a matriz de gradiente dos pesos $dW$ com o mesmo formato $m \times n$ de $W$, e `transposeM` combinado com `matVec` calcula $W^T \cdot dz$ para propagar o erro às camadas anteriores.
- **`zipNetworkWith`, `mapNetwork`, `addNetworks`, `scaleNetwork` e `zeroNetwork`**: abstraem operações sobre todos os pesos e biases da rede inteira, permitindo acumular e tirar média de gradientes de um minibatch em `DQN.Backprop` e executar as atualizações do SGD e do otimizador Adam (sobre `adamM` e `adamV`) em `DQN.Optimizer`.

## Perguntas prováveis da arguição

1. **Pergunta:** Por que não dá para usar Q-Learning tabular no CartPole?
   **Resposta:** Porque o estado do CartPole é formado por 4 números reais contínuos (posição, velocidade, ângulo e velocidade angular). Uma tabela precisaria de infinitas linhas para representar todos os estados possíveis. Mesmo discretizando os intervalos, o número de combinações cresce exponencialmente e a tabela não generaliza o aprendizado entre estados muito próximos.

2. **Pergunta:** Qual é o papel da equação de Bellman no treinamento do agente?
   **Resposta:** Ela relaciona o valor de um par $(s, a)$ à recompensa imediata $r$ mais o melhor valor futuro descontado $\gamma \max_{a'} Q(s', a')$. No Q-Learning e no DQN, esse lado direito vira o *alvo* ($y$) que queremos que a estimativa atual $Q(s, a)$ alcance. No nosso DQN, esse alvo é calculado usando a *target network* para dar estabilidade ao treino.

3. **Pergunta:** Por que o `outer` (produto externo) em `DQN.Linalg` precisa devolver uma matriz com `length a` linhas e `length b` colunas, e não o contrário?
   **Resposta:** Porque no backpropagation calculamos o gradiente dos pesos de uma camada como `outer dz entrada`, onde `dz` tem tamanho $m$ (número de neurônios de saída da camada) e `entrada` tem tamanho $n$ (número de entradas). Como a matriz de pesos `weights` da camada tem dimensão $m \times n$, o gradiente `dW` precisa ter exatamente a mesma dimensão $m \times n$ para ser somado/subtraído dos pesos na atualização.

4. **Pergunta:** Por que implementamos `zipNetworkWith` e `zeroNetwork` dentro de `DQN.Linalg` se elas operam sobre `Network` e não apenas sobre vetores e matrizes?
   **Resposta:** Porque no projeto o mesmo tipo `Network` representa os pesos da rede, os gradientes calculados pelo backprop e os estados internos `adamM` e `adamV` do otimizador Adam. Centralizar operações elemento a elemento sobre `Network` em `DQN.Linalg` evita repetir código de iteração sobre camadas e matrizes tanto no `DQN.Backprop` quanto no `DQN.Optimizer`.

## Referências

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Sutton, R. S.; Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. 2ª ed. MIT Press.