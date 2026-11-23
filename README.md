# Deep Q-Learning em Haskell

Implementação **do zero, em Haskell puro**, de um agente **Deep Q-Learning
(DQN)** que aprende a equilibrar uma haste em cima de um carrinho (o problema
**CartPole**). Sem bibliotecas de machine learning: a rede neural, o
backpropagation, o otimizador Adam, o replay buffer e a física do ambiente
foram escritos pelo grupo.

Trabalho em grupo: tema 3.4 (Deep Q-Learning) do projeto *Aprendizado por
Reforço, Blockchain e Teste Baseado em Modelos*.

## Integrantes e divisão

Veja [`docs/DIVISAO.md`](docs/DIVISAO.md).

## Como rodar

### 1. Instalar o Haskell (uma vez)

Instale o **GHCup** seguindo as instruções de <https://www.haskell.org/ghcup/>
(Mac, Linux e Windows). Depois:

```bash
ghcup install ghc 9.6 --set
ghcup install cabal --set
cabal update
```

### 2. Compilar, testar e executar

```bash
cabal build          # compila (na primeira vez baixa as dependências)
cabal test           # roda os testes
cabal run dqn        # treina o agente (semente 42, 400 episódios)
cabal run dqn -- 7 300   # semente 7, 300 episódios
```

O treino leva de 1 a 3 minutos. Ao final, o programa:

- salva o histórico em `results/training.csv`;
- mostra no terminal o gráfico da média móvel da recompensa;
- compara a política aleatória, a rede final e a melhor rede encontrada.

## Estrutura

```
src/DQN/
  Types.hs       tipos compartilhados (o contrato do projeto)
  Env.hs         Parte 1: ambiente CartPole
  Linalg.hs      Parte 2: álgebra linear e operações sobre redes
  Network.hs     Parte 3: rede neural (inicialização e forward)
  Backprop.hs    Parte 4: perda de Huber e backpropagation
  Replay.hs      Parte 5: experience replay
  Policy.hs      Parte 6: política ε-greedy
  Agent.hs       Parte 7: algoritmo DQN (target network, loop de treino)
  Experiment.hs  Parte 8: CSV, avaliação e gráfico
  Optimizer.hs   Parte 9: SGD e Adam
app/Main.hs      programa principal (Parte 8)
test/DQN/        testes de cada módulo (hspec)
docs/teoria/     parte teórica, uma seção por parte
```

## Para quem vai contribuir

1. Leia o [`AGENTS.md`](AGENTS.md): tem as regras e a especificação da sua parte.
2. Siga o [`docs/GUIA-GIT.md`](docs/GUIA-GIT.md) para criar sua branch, commitar e abrir o PR.
3. Se for usar IA, veja a seção "Usando IA" do guia.
