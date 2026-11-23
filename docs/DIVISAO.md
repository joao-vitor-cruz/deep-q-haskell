# Divisão do trabalho

Cada integrante faz **pelo menos 2 commits da própria conta**: um de código
(módulo + testes) e um de teoria (`docs/teoria/`).

| Parte | Módulo | Teoria | Integrante | Usuário GitHub | Onda |
|---|---|---|---|---|---|
| 1 | `DQN.Env` | 01: Fundamentos de RL | | | 1 |
| 2 | `DQN.Linalg` | 02: Do Q-Learning ao DQN | | | 1 |
| 3 | `DQN.Network` | 03: Redes neurais | | | 2 |
| 4 | `DQN.Backprop` | 04: Perda e backpropagation | | | 3 |
| 5 | `DQN.Replay` | 05: Experience replay | | | 1 |
| 6 | `DQN.Policy` | 06: Exploração (ε-greedy) | | | 3 |
| 7 | `DQN.Agent` | 07: Arquitetura do DQN | | | 4 |
| 8 | `DQN.Experiment` + `app/Main.hs` | 08: Experimento e resultados | | | 5 |
| 9 | `DQN.Optimizer` | 09: Otimização, limitações e aplicações | | | 2 |

**Ondas:** todo mundo pode programar ao mesmo tempo, mas o PR só passa na CI
depois que as partes de que ele depende entraram na `main` (AGENTS.md, seção 5).
Quem é da onda 1 deve abrir o PR o quanto antes, porque os outros dependem disso.

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
