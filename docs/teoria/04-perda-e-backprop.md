# Função de perda, alvo de Bellman e backpropagation

> **Parte 4** · Responsável: João Vítor · Código relacionado: `DQN.Backprop`

## Conceito

### 1. O que a rede faz e onde a Parte 4 entra

A rede recebe os 4 números do estado do CartPole (posição e velocidade do
carrinho, ângulo e velocidade angular da haste) e devolve 2 números: uma
previsão de quantos pontos o agente ainda vai fazer, a partir do estado
atual, se empurrar o carrinho para a direita, e o mesmo se empurrar para a
esquerda. Esses números são os Q-valores. O agente escolhe a ação com o
maior número.

No começo do treino essas previsões estão erradas, porque os pesos da rede
são sorteados (`initNetwork`, Parte 3) e ela ainda não aprendeu nada. Eles
não começam todos iguais porque, se começassem, todos os neurônios
calculariam a mesma coisa, receberiam a mesma correção e seria como ter um
neurônio só.

Treinar a rede é repetir este ciclo milhares de vezes:

```
1. A rede prevê Q(s, a)                             Parte 3 (forward)
2. Calcula-se o alvo y                              Parte 7 (tdTarget)
3. Mede-se o erro com a perda de Huber              Parte 4  ← esta parte
4. O backprop calcula o gradiente de cada peso      Parte 4  ← esta parte
5. O otimizador atualiza os pesos                   Parte 9 (Adam)
```

A Parte 4 mede o erro da previsão em relação ao alvo (a perda) e calcula o
gradiente, ou seja, quanto e para que lado cada peso deveria mudar. Ela não
faz a previsão (Parte 3), não calcula o alvo (Parte 7) e não atualiza os
pesos (Parte 9).

### 2. O alvo de Bellman e o erro TD

Ninguém diz à rede qual é o Q-valor certo, então o alvo é construído a cada
passo. Ele junta a recompensa que o agente de fato ganhou (`r`) com um
palpite de quantos pontos ainda dá para fazer a partir do próximo estado
(`max Q(s', a')`, calculado pela rede target). Esse palpite é multiplicado
pelo fator de desconto `γ = 0,99`, porque pontos no futuro valem um pouco
menos.

É como o GPS: ele diz que a viagem leva 30 minutos. Depois de 5 minutos
dirigindo, diz que faltam 22. A nova estimativa para a viagem toda é
5 + 22 = 27, e o GPS percebe que errou por 3. Os 5 minutos são realidade (o
`r`), os 22 são palpite (o `γ · max Q(s', a')`) e o 27 é o alvo (`y`). O
alvo é realidade mais palpite, não só realidade.

**A recompensa `r`.** É o número que o ambiente entrega depois de cada
ação. No CartPole, `r = 1` em todo passo em que a haste está em pé,
inclusive no passo em que ela cai. Assim, o total do episódio é igual ao
número de passos em pé. Quem entrega o `r` é o `stepEnv` (Parte 1).

**Quando a haste cai.** Não há mais pontos para ganhar, e o alvo é só
`y = r = 1`. Esse alvo baixo, comparado com os ~20 de um passo normal, é o
que ensina a rede que cair é ruim. A punição vem de perder os pontos
futuros, e não de uma recompensa negativa.

**O erro TD** (diferença temporal) é o quanto a previsão ficou longe do
alvo: `d = previsão − alvo`. Negativo quer dizer que a rede foi pessimista e
vai ser corrigida para cima. Positivo quer dizer que foi otimista.

O alvo chega pronto para a Parte 4 e é tratado como **constante**: o
gradiente não passa por ele.

### 3. A perda: Huber vs. erro quadrático

A função de perda transforma o erro TD num número que mede o quanto a rede
está indo mal. Ele é sempre maior ou igual a zero e só vale zero quando a
rede acerta. Não dá para usar o erro TD direto, porque minimizá-lo levaria a
rede a buscar erros cada vez mais negativos, que também são erros.

O erro quadrático (`½·d²`) tem um problema: a força da correção (o
gradiente) cresce junto com o erro, sem limite. Um erro grande faz a rede
mudar tudo, aleatorizando-a em vez de lapidá-la. No DQN isso é pior, porque
o alvo é parte realidade e parte palpite de uma rede que no começo não sabe
nada. Um erro grande pode ser culpa do alvo, e mesmo assim a rede reagiria
com força máxima.

A perda de Huber é igual à quadrática quando o erro é pequeno (`|d| ≤ 1`) e
cresce em linha reta quando é grande (`|d| − 0,5`). Assim, o gradiente fica
limitado ao intervalo [−1, 1]: a perda continua crescendo, mas a força da
correção nunca passa de 1. O artigo da DeepMind (Mnih et al., 2015) faz o
mesmo e descreve isso como cortar o erro em [−1, 1].

### 4. Gradiente descendente

É como descer uma montanha de olhos vendados. A altura do terreno é a perda
(o quanto a rede erra), a posição onde estou são os pesos atuais e o vale
são os pesos com a menor perda. Não dá para ver o vale, mas dá para sentir a
inclinação do chão, que é o gradiente.

O gradiente de um peso diz se aumentá-lo faz a perda subir ou descer, e
quanto. Andamos sempre no sentido contrário: com gradiente positivo, o peso
diminui; com negativo, o peso aumenta.

A taxa de aprendizado (`lr`, 0,001 no projeto) é o tamanho do passo. Grande
demais, a rede muda tanto que passa por cima do vale. Pequena demais, demora
muito a chegar.

O DQN não usa uma experiência só: sorteia 32 do replay buffer e tira a média
dos gradientes, para o passo ser mais confiável.

### 5. Backpropagation e regra da cadeia

A rede tem duas camadas em sequência: `4 entradas → 64 neurônios (ReLU) → 2
saídas (linear)`. Um peso da última camada afeta os Q-valores diretamente.
Um peso da primeira camada afeta indiretamente: muda um neurônio oculto, que
muda a última camada, que muda os Q-valores. O backpropagation calcula o
gradiente de todos esses pesos.

Uma analogia é a linha de produção de um bolo:

```
Ana (camada 1) → Caio (camada 2) → bolo (Q-valores) → provador (perda)
```

O provador diz o quanto o bolo errou. Para dividir a culpa, começa-se pelo
fim: primeiro calcula-se a culpa de cada ajuste do Caio, que mexeu direto no
bolo. Depois essa culpa é repassada para a Ana, na medida em que a massa que
ela entregou influenciou o que o Caio fez. Por isso o nome: o erro nasce na
saída e é **propagado para trás**, camada por camada.

- **Regra da cadeia:** é o nome matemático de "repassar a culpa
  multiplicando". Se a engrenagem A gira 2 vezes mais rápido que B, e B 3
  vezes mais que C, então A gira 2 × 3 = 6 vezes mais que C. O efeito de um
  peso na perda é o produto dos efeitos ao longo do caminho.
- **Por que de trás para frente:** a culpa da camada 1 depende da culpa da
  camada 2. Começando pelo fim, a culpa da camada 2 é calculada uma vez só e
  reaproveitada para todos os neurônios da camada 1, sem refazer o caminho
  inteiro para cada peso.
- **A ReLU que "desliga":** se o valor de um neurônio oculto antes da ReLU
  (`z`) for menor ou igual a 0, a saída dele é 0. Ele não influenciou o
  resultado e não recebe culpa: os pesos dele têm gradiente zero nessa
  rodada. Isso é a derivada da ReLU, que vale 1 se `z > 0` e 0 caso
  contrário. A última camada não tem ReLU (é linear), por isso não
  multiplica por essa derivada.
- **Só a ação tomada leva culpa:** só existe alvo para a ação que o agente
  de fato escolheu. O gradiente da outra saída começa em zero.

O backprop devolve um objeto com o mesmo formato da rede (uma matriz e um
vetor por camada), mas em vez de pesos ele guarda o gradiente de cada peso e
de cada bias. É esse pacote que vai para o otimizador.

Para fazer o caminho de volta, o backprop precisa saber o que entrou em cada
camada e o valor de `z` de cada neurônio. Esses valores são guardados
durante a previsão pelo `forwardTrace` (Parte 3).

### 6. Verificação numérica do gradiente

Como saber se o backprop está certo? Um erro de sinal ou de índice não faz o
programa travar: ele roda, mas a rede não aprende. Por isso o backprop é
conferido com um método mais simples e lento, que não tem como errar do
mesmo jeito: **medir o gradiente "na mão"**.

A ideia vem da própria definição de derivada. Pega-se um peso `w`,
aumenta-se ele um pouquinho (`h = 0,000001`) e mede-se o quanto a perda
mudou:

```
gradiente aproximado = (L(w + h) − L(w)) / h
```

Se o resultado bater com o gradiente que o backprop calculou (com diferença
menor que ~0,0001), o backprop está correto. O teste faz isso para vários
pesos de uma rede pequena (por exemplo, `[4, 5, 3, 2]`).

Esse método não é usado no treino porque exige uma previsão completa da rede
para **cada peso**. Com milhares de pesos, seriam milhares de previsões por
passo de treino. O backprop calcula todos os gradientes de uma vez. A
verificação numérica serve só para testar.

## Fórmulas

**Alvo de Bellman:**

```
y = r                            se o episódio terminou (done)
y = r + γ · max_a' Q(s', a')     caso contrário
```

- `r`: recompensa do passo (1 no CartPole);
- `γ`: fator de desconto (0,99). Não confundir com `y`, que é o alvo;
- `s'`: próximo estado;
- `max_a' Q(s', a')`: o maior Q-valor do próximo estado, segundo a rede target.

**Erro TD:** `d = Q(s, a) − y`

**Perda de Huber** (δ = 1):

```
L(d) = ½ · d²       se |d| ≤ 1
L(d) = |d| − ½      se |d| > 1
```

**Derivada da Huber** em relação à previsão: `d` cortado no intervalo
[−1, 1].

**Gradiente descendente** (aplicado pela Parte 9, não por esta):

```
peso_novo = peso_antigo − lr · gradiente
```

**Backpropagation**, para cada camada, da última para a primeira (`dA`
começa como o gradiente da perda em relação à saída da rede):

```
dz     = dA                         última camada (linear)
dz     = dA ⊙ ReLU'(z)              camadas ocultas
dW     = outer dz entrada           (mesmo formato de W)
db     = dz
dA_ant = Wᵀ · dz                    vira o dA da camada anterior
```

- `⊙`: produto elemento a elemento;
- `z = W·x + b`: valor da camada antes da ativação;
- `ReLU'(z)`: 1 se `z > 0`, 0 caso contrário.

**Minibatch:** gradiente e perda finais são a média sobre as `N` amostras.

**Verificação numérica:** `(L(w + h) − L(w)) / h`, com `h = 1e-6`.

## Exemplo

### Huber × quadrática

Com `r = 1`, `γ = 0,99` e a rede target dizendo `[18,2; 20,0]` para o
próximo estado:

```
y = 1 + 0,99 · 20,0 = 1 + 19,8 = 20,8
```

Se a rede previu 19,5, o erro TD é `d = 19,5 − 20,8 = −1,3`.

| | Perda | Gradiente |
|---|---|---|
| Quadrática | ½ · 1,69 = 0,845 | −1,3 |
| Huber | 1,3 − 0,5 = 0,8 | −1 (cortado) |

Com `d = −4`, a quadrática daria gradiente −4, e a Huber continua em −1. Se
a haste tivesse caído nesse passo, o alvo seria só `y = 1`.

### Backprop numa rede pequena

Rede `[1, 2, 2]`: 1 entrada, 2 neurônios ocultos (ReLU), 2 saídas (linear),
biases zero.

```
W1 = [[ 2],         W2 = [[1,   3],
      [-1]]               [0,5, 1]]
x  = [1]
```

**Previsão (forward):**

```
z1 = W1·x  = [2, −1]
a1 = ReLU(z1) = [2, 0]          (o neurônio 2 ficou desligado)
Q  = W2·a1 = [1·2 + 3·0, 0,5·2 + 1·0] = [2, 1]
```

Ação tomada `a = 0` e alvo `y = 3,5`. Erro `d = 2 − 3,5 = −1,5`, perda de
Huber `1,5 − 0,5 = 1,0`. O gradiente em relação à saída é `[−1, 0]`: −1
(cortado) na ação tomada e 0 na outra.

**Caminho de volta (backward):**

```
Camada 2 (linear):
  dz2 = [−1, 0]
  dW2 = outer dz2 a1 = [[−2, 0], [0, 0]]
  db2 = [−1, 0]
  dA1 = W2ᵀ · dz2 = [−1, −3]

Camada 1 (ReLU):
  ReLU'(z1) = [1, 0]
  dz1 = dA1 ⊙ ReLU'(z1) = [−1, 0]
  dW1 = outer dz1 x = [[−1], [0]]
  db1 = [−1, 0]
```

O neurônio 2 da camada oculta, desligado pela ReLU, ficou com gradiente
zero.

**Conferência numérica:** aumentando `W1[0][0]` de 2 para `2 + h`, a saída
`Q[0]` vira `2 + h`, o erro vira `−1,5 + h` e a perda cai `h`. Então o
gradiente é −1, igual ao `dW1[0][0]` do backprop.

**Um passo de gradiente descendente** com `lr = 0,1`, fazendo
`novo = antigo − 0,1 · gradiente` em cada peso e bias:

```
W1[0][0]: 2 → 2,1      b1[0]: 0 → 0,1
W2[0][0]: 1 → 1,2      b2[0]: 0 → 0,1
(os outros têm gradiente zero e não mudam)
```

Nova previsão: `z1[0] = 2,1 · 1 + 0,1 = 2,2` e
`Q[0] = 1,2 · 2,2 + 0,1 = 2,74`. Saiu de 2 e ficou mais perto do alvo 3,5.

## Como isso aparece no nosso código

Módulo `DQN.Backprop` (`src/DQN/Backprop.hs`):

| Conceito | Função | O que faz |
|---|---|---|
| Perda de Huber | `huberLoss previsao alvo` | `½·d²` se `\|d\| ≤ 1`, senão `\|d\| − ½`. |
| Derivada da Huber | `huberLossDerivative previsao alvo` | `d` cortado em [−1, 1]. |
| Backpropagation | `backprop rede x dSaida` | Percorre as camadas da última para a primeira e devolve um `Network` com o gradiente de cada peso e bias. |
| Gradiente de uma amostra | `sampleGradient rede x a y` | Monta o gradiente da saída (zero em tudo, menos na ação `a`), chama `backprop` e devolve também a perda. |
| Média do minibatch | `batchGradient rede amostras` | Média dos gradientes e das perdas das amostras. |

Ligações com outras partes:

- **Parte 2 (`DQN.Linalg`):** `outer`, `matVec`, `transposeM`, `hadamard`,
  `addNetworks` e `scaleNetwork` fazem as contas de vetor, matriz e rede.
- **Parte 3 (`DQN.Network`):** `forwardTrace` guarda a entrada e o `z` de
  cada camada, e `reluDerivative` dá a derivada da ReLU.
- **Parte 7 (`DQN.Agent`):** `trainStep` calcula os alvos com `tdTarget`
  (rede target) e chama `batchGradient`.
- **Parte 9 (`DQN.Optimizer`):** `adamStep` recebe o gradiente e atualiza os
  pesos.

Testes em `test/DQN/BackpropSpec.hs`: verificação numérica do gradiente,
gradiente zero quando previsão = alvo, `huberLoss` e `huberLossDerivative`
dentro e fora de [−1, 1], e `batchGradient` de uma amostra igual a
`sampleGradient`.

## Perguntas prováveis da arguição

1. **Pergunta:** O seu módulo atualiza os pesos da rede?
   **Resposta:** Não. Ele só calcula a perda e o gradiente. Quem aplica o
   passo `peso − lr · gradiente` (no nosso caso, com Adam) é o
   `DQN.Optimizer`, da Parte 9.

2. **Pergunta:** De onde vem o alvo, se ninguém diz o Q-valor certo?
   **Resposta:** Da equação de Bellman: `y = r + γ · max Q(s', a')`. Junta a
   recompensa real do passo com um palpite da rede target sobre o próximo
   estado. É como o GPS que recalcula depois de alguns minutos dirigindo.

3. **Pergunta:** O que acontece com o alvo quando a haste cai?
   **Resposta:** Vira só `y = r = 1`, porque não há mais pontos futuros.
   Somar o palpite nesse caso contaria pontos que nunca vão chegar. Esse
   alvo baixo é o que ensina a rede que cair é ruim.

4. **Pergunta:** Por que Huber e não erro quadrático?
   **Resposta:** Com erro quadrático, a força da correção cresce junto com o
   erro. No DQN o alvo é parcialmente um palpite, então um erro grande pode
   ser culpa do alvo. A Huber limita o gradiente a [−1, 1], e o treino fica
   estável.

5. **Pergunta:** Com erro 50, a perda de Huber vale 1?
   **Resposta:** Não. A perda vale `50 − 0,5 = 49,5`. O que vale no máximo 1
   é o gradiente.

6. **Pergunta:** Por que a perda só olha a saída da ação escolhida?
   **Resposta:** Só temos recompensa observada para a ação que de fato foi
   tomada. Para a outra não existe alvo, então o gradiente dela é zero.

7. **Pergunta:** O que é o backpropagation e por que vai de trás para frente?
   **Resposta:** É a regra da cadeia aplicada da saída para a entrada: o erro
   é repassado camada por camada, e cada uma calcula o gradiente dos próprios
   pesos. Vai de trás para frente porque a culpa de uma camada depende da
   culpa da camada seguinte. Assim cada conta é feita uma vez só.

8. **Pergunta:** Por que multiplicar pela derivada da ReLU nas camadas
   ocultas, e não na última?
   **Resposta:** Só as camadas ocultas têm ReLU. Um neurônio com `z ≤ 0` foi
   desligado, não influenciou a saída e não recebe gradiente. A última
   camada é linear, e a derivada dela é 1.

9. **Pergunta:** Como vocês sabem que o backprop está certo?
   **Resposta:** Pela verificação numérica: aumentamos um peso em `h = 1e-6`,
   medimos `(L(w + h) − L(w)) / h` e comparamos com o gradiente do backprop.
   Se a diferença for menor que ~1e-4, está certo.

10. **Pergunta:** Por que não usar a verificação numérica no treino, já que é
    mais simples?
    **Resposta:** Porque precisa de uma previsão completa da rede para cada
    peso. Com milhares de pesos, o treino ficaria milhares de vezes mais
    lento. O backprop calcula todos os gradientes de uma vez.

11. **Pergunta:** Por que os pesos começam aleatórios e não zerados?
    **Resposta:** Se todos fossem iguais, todos os neurônios calculariam a
    mesma coisa e receberiam a mesma correção. Continuariam iguais para
    sempre. O sorteio faz cada neurônio aprender algo diferente.

## Referências

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Sutton, R. S.; Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. 2ª ed. MIT Press.
- Rumelhart, D. E.; Hinton, G. E.; Williams, R. J. (1986). *Learning representations by back-propagating errors*. Nature, 323, 533–536.
- Huber, P. J. (1964). *Robust estimation of a location parameter*. Annals of Mathematical Statistics, 35(1), 73–101.
