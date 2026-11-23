# Fundamentos de Aprendizado por Reforço

> **Parte 1** · Responsável: _(Raphael L. M. Resende)_ · Código relacionado: `DQN.Env`

## O que esta seção precisa cobrir

- [x] O que é aprendizado por reforço e como difere do aprendizado supervisionado
- [x] Agente, ambiente, estado, ação, recompensa e episódio
- [x] Política (π) e função de valor (V e Q)
- [x] Processo de Decisão de Markov (MDP) e fator de desconto γ
- [x] O CartPole como problema de RL: qual é o estado, quais são as ações, qual é a recompensa e quando o episódio termina

## Conceito

O **Aprendizado por Reforço** (*Reinforcement Learning* — RL) é um paradigma de aprendizado de máquina em que um **agente** aprende a tomar decisões sequenciais interagindo diretamente com um **ambiente** por tentativa e erro. Diferentemente do **aprendizado supervisionado** — no qual existe um conjunto fixo de exemplos rotulados indicando a resposta correta para cada entrada —, no RL ninguém fornece a ação ideal em cada instante. O agente recebe apenas um sinal numérico de **recompensa** (*reward*) após agir e precisa descobrir sozinho quais sequências de ações maximizam a recompensa acumulada ao longo do tempo.

Essa interação ocorre em passos discretos de tempo $t = 0, 1, 2, \dots$:

1. O ambiente apresenta ao agente um **estado** $s_t \in \mathcal{S}$, que descreve a situação atual do sistema.
2. O agente escolhe uma **ação** $a_t \in \mathcal{A}$ seguindo sua **política** $\pi$, que mapeia estados em ações (de forma determinística $a_t = \pi(s_t)$ ou probabilística $\pi(a_t \mid s_t)$).
3. O ambiente executa a transição física/lógica, devolvendo o próximo estado $s_{t+1}$ e uma **recompensa** escalar $r_{t+1} \in \mathbb{R}$.
4. Esse ciclo se repete até atingir um **estado terminal** (ou um limite máximo de passos), encerrando um **episódio**.

Para avaliar se vale a pena estar em um estado ou executar uma ação, definem-se duas funções de valor:

- **Função de valor de estado $V^\pi(s)$:** retorno esperado partindo do estado $s$ e seguindo a política $\pi$.
- **Função de valor de ação $Q^\pi(s, a)$:** retorno esperado ao executar a ação $a$ no estado $s$ e, a partir daí, seguir a política $\pi$. O algoritmo **Deep Q-Learning (DQN)** usa uma rede neural para aproximar justamente a função $Q^*(s, a)$ ótima.

Todo esse processo é formalizado matematicamente como um **Processo de Decisão de Markov (MDP)**, definido pela tupla $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$, onde $P(s' \mid s, a)$ é a dinâmica de transição (que satisfaz a *propriedade de Markov*: o futuro depende apenas do estado atual $s_t$ e da ação $a_t$, e não de todo o histórico anterior), $R$ é a função de recompensa e $\gamma \in [0, 1)$ é o **fator de desconto**.

No problema específico do **CartPole**, um carrinho se move sobre um trilho horizontal de atrito desprezível carregando uma haste presa por um pivô:

- **Estado ($\mathcal{S} \subset \mathbb{R}^4$):** vetor contínuo de 4 variáveis reais $(x, \dot{x}, \theta, \dot{\theta})$, representando posição do carrinho, velocidade linear, ângulo da haste em relação à vertical e velocidade angular.
- **Ações ($\mathcal{A}$):** conjunto discreto com 2 opções — empurrar o carrinho para a esquerda ($F = -10\text{ N}$) ou para a direita ($F = +10\text{ N}$).
- **Recompensa ($R$):** $+1.0$ para cada passo de tempo ($\tau = 0.02\text{ s}$) em que o sistema permanece válido, inclusive no passo em que ocorre a falha. Assim, maximizar o retorno equivale a manter a haste equilibrada pelo maior tempo possível.
- **Término do episódio:** ocorre quando o carrinho ultrapassa os limites do trilho ($|x| > 2.4\text{ m}$) ou a haste inclina demais ($|\theta| > 12^\circ$).

## Fórmulas

**Retorno descontado ($G_t$):** soma das recompensas futuras ponderadas pelo fator de desconto $\gamma \in [0, 1)$:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots$$

- Quando $\gamma \to 0$, o agente torna-se "míope", priorizando apenas recompensas imediatas.
- Quando $\gamma \to 1$ (no nosso projeto, $\gamma = 0.99$), recompensas futuras têm quase o mesmo peso que as imediatas, incentivando o agente a equilibrar a haste no longo prazo.

**Equação de Bellman para a função $Q$ ótima ($Q^*$):**

$$Q^*(s, a) = \mathbb{E}\left[ r_{t+1} + \gamma \max_{a' \in \mathcal{A}} Q^*(s_{t+1}, a') \;\middle|\; s_t = s, a_t = a \right]$$

**Dinâmica física do CartPole (passo de tempo $\tau = 0.02\text{ s}$):**

$$\text{temp} = \frac{F + m_p \cdot l \cdot \dot{\theta}^2 \cdot \sin\theta}{M}, \qquad M = m_c + m_p$$

$$\ddot{\theta} = \frac{g \sin\theta - \cos\theta \cdot \text{temp}}{l \left(\frac{4}{3} - \frac{m_p \cos^2\theta}{M}\right)}, \qquad \ddot{x} = \text{temp} - \frac{m_p \cdot l \cdot \ddot{\theta} \cdot \cos\theta}{M}$$

**Integração de Euler explícita:**

$$x_{t+1} = x_t + \tau \dot{x}_t, \qquad \dot{x}_{t+1} = \dot{x}_t + \tau \ddot{x}_t$$

$$\theta_{t+1} = \theta_t + \tau \dot{\theta}_t, \qquad \dot{\theta}_{t+1} = \dot{\theta}_t + \tau \ddot{\theta}_t$$

## Exemplo

Considere um estado inicial em que o carrinho está parado no centro, mas a haste está levemente inclinada para a direita:

$$s_0 = (x = 0.0\text{ m},\; \dot{x} = 0.0\text{ m/s},\; \theta = 0.05\text{ rad},\; \dot{\theta} = 0.0\text{ rad/s})$$

O agente aplica a ação $a_0 = \text{PushRight}$ ($F = +10.0\text{ N}$), usando $g = 9.8$, $m_c = 1.0$, $m_p = 0.1$, $M = 1.1$, $l = 0.5$ e $\tau = 0.02$:

1. $\text{temp} = \frac{10.0 + 0.05 \cdot 0^2 \cdot \sin(0.05)}{1.1} = \frac{10.0}{1.1} \approx 9.0909\text{ m/s}^2$
2. $\ddot{\theta} = \frac{9.8 \cdot \sin(0.05) - \cos(0.05) \cdot 9.0909}{0.5 \cdot \left(\frac{4}{3} - \frac{0.1 \cdot \cos^2(0.05)}{1.1}\right)} \approx \frac{0.4898 - 9.0795}{0.6213} \approx -13.825\text{ rad/s}^2$
3. $\ddot{x} = 9.0909 - \frac{0.05 \cdot (-13.825) \cdot \cos(0.05)}{1.1} \approx 9.7185\text{ m/s}^2$
4. Atualizando por Euler ($\tau = 0.02$):
   - $x_1 = 0.0 + 0.02 \cdot 0.0 = 0.0\text{ m}$
   - $\dot{x}_1 = 0.0 + 0.02 \cdot 9.7185 \approx 0.1944\text{ m/s}$
   - $\theta_1 = 0.05 + 0.02 \cdot 0.0 = 0.05\text{ rad}$
   - $\dot{\theta}_1 = 0.0 + 0.02 \cdot (-13.825) \approx -0.2765\text{ rad/s}$

Como $|x_1| = 0.0 \le 2.4$ e $|\theta_1| = 0.05 \le 0.2094\text{ rad}$ ($12^\circ$), o episódio **não** terminou (`done = False`) e o agente recebe $r_1 = 1.0$. Observe que empurrar para a direita gerou aceleração angular negativa ($\ddot{\theta} < 0$), corrigindo a inclinação da haste.

## Como isso aparece no nosso código

Os elementos do MDP são definidos como tipos imutáveis em `src/DQN/Types.hs` e simulados de forma puramente funcional em `src/DQN/Env.hs`:

- **Estado ($s_t$):** representado pelo registro `CartState` (`cartPosition`, `cartVelocity`, `poleAngle`, `poleAngularVelocity`). A função `stateToVector` converte esse registro em um `Vector` (`[Double]`) na ordem fixa `[x, xDot, theta, thetaDot]` para alimentar a camada de entrada da rede neural.
- **Início de episódio ($s_0$):** implementado por `resetEnv :: StdGen -> (CartState, StdGen)`, que sorteia cada uma das 4 variáveis uniformemente em $[-0.05, 0.05]$ e encadeia o gerador `StdGen` explicitamente, sem efeitos colaterais (`IO`).
- **Ação ($a_t$):** representada pelo tipo algébrico `Action = PushLeft | PushRight`, mapeado aos índices `0` e `1` da saída da rede por `actionToIndex` e `indexToAction`.
- **Transição do ambiente ($P$ e $R$):** implementada por `stepEnv :: CartState -> Action -> StepResult`, que calcula as acelerações $\ddot{\theta}$ e $\ddot{x}$, integra as equações de Euler explícita e retorna o `StepResult` contendo `nextState`, `reward = 1.0` e `done`.
- **Condição de parada:** implementada por `isTerminal :: CartState -> Bool`, que verifica $|x| > 2.4$ ou $|\theta| > 12\pi / 180$.

## Perguntas prováveis da arguição

1. **Pergunta:** Por que não usamos uma tabela Q tradicional no CartPole em vez de uma rede neural?
   **Resposta:** Porque o estado `CartState` é composto por quatro números reais contínuos (`Double`). O espaço de estados é infinito (ou astronômico se discretizado com precisão), tornando inviável armazenar e visitar todos os pares $(s, a)$ em uma tabela. A rede neural aproxima $Q(s, a)$ e generaliza entre estados próximos.

2. **Pergunta:** Por que `resetEnv` recebe e devolve um `StdGen` em vez de gerar números aleatórios diretamente com `randomIO`?
   **Resposta:** Porque o módulo `DQN.Env` é 100% puro e livre de `IO`. Passar o `StdGen` explicitamente garante transparência referencial e reprodutibilidade: usando a mesma semente inicial (como `mkStdGen 42`), toda a sequência de estados iniciais e episódios é determinística e testável.

3. **Pergunta:** Qual erro aconteceria se atualizássemos a posição $x'$ usando a velocidade nova $\dot{x}'$ em vez da antiga $\dot{x}$?
   **Resposta:** Estaríamos trocando o método de Euler explícito pelo método de Euler semi-implícito, alterando a trajetória numérica da simulação e fazendo o ambiente divergir da especificação padrão do `CartPole-v1`.

4. **Pergunta:** Por que o limite de ângulo no código é escrito como `12.0 * pi / 180.0` em vez de `12.0`?
   **Resposta:** Porque `poleAngle` e as funções trigonométricas `sin` e `cos` da biblioteca padrão do Haskell trabalham em radianos. Comparar diretamente com `12.0` radianos permitiria quase duas voltas completas da haste antes de encerrar o episódio.

## Referências

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Sutton, R. S.; Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. 2ª ed. MIT Press.