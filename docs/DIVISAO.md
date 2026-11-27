# Divisão do trabalho

Cada integrante faz **pelo menos 2 commits da própria conta**: um de código
(módulo + testes) e um de teoria (`docs/teoria/`).

| Parte | Módulo | Teoria | Integrante | Usuário GitHub | Onda |
|---|---|---|---|---|---|
| 1 | `DQN.Env` | 01: Fundamentos de RL | Raphael | | 1 |
| 2 | `DQN.Linalg` | 02: Do Q-Learning ao DQN | Bruno | | 1 |
| 3 | `DQN.Network` | 03: Redes neurais | Felipe | | 2 |
| 4 | `DQN.Backprop` | 04: Perda e backpropagation | João Vítor | | 3 |
| 5 | `DQN.Replay` | 05: Experience replay | João Gabriel | | 1 |
| 6 | `DQN.Policy` | 06: Exploração (ε-greedy) | Guilherme | | 3 |
| 7 | `DQN.Agent` | 07: Arquitetura do DQN | Gabriel Fagundes | | 4 |
| 8 | `DQN.Experiment` + `app/Main.hs` | 08: Experimento e resultados | Lucas | | 5 |
| 9 | `DQN.Optimizer` | 09: Otimização, limitações e aplicações | Helena | | 2 |

**Ondas:** todo mundo pode programar ao mesmo tempo, mas o PR só passa na CI
depois que as partes de que ele depende entraram na `main` (AGENTS.md, seção 5).
Quem é da onda 1 deve abrir o PR o quanto antes, porque os outros dependem disso.

## Slides

Deck da apresentação (20 slides, um tema por slide, com notas do apresentador
indicando a parte e o responsável):
https://claude.ai/artifact/2GvmRNxaREKAw6w1qNaf2x

- Dá para baixar como PowerPoint (.pptx) ou PDF pelo próprio deck.
- O slide de **Resultados** tem campos `[__]` para preencher depois que a
  Parte 8 entrar e o treino rodar (`cabal run dqn -- 42 400`).
- Cada integrante revisa o slide da sua parte.

## Prazos

| Marco | Data |
|---|---|
| Onda 1 na `main` (Env, Linalg, Replay) | |
| Ondas 2 e 3 na `main` | |
| Onda 4 (Agent) na `main` | |
| Onda 5 (Main) e projeto rodando | |
| Teoria de todas as partes | |
| Slides prontos | |
| Ensaio da apresentação | |
| Apresentação | |
