-- |
-- Module      : DQN.Policy
-- Description : Parte 6 — exploração ε-greedy.
--
-- Decide qual ação o agente toma em cada passo. A política mistura dois
-- comportamentos: /aproveitar/ o que a rede já aprendeu (ação gulosa) e
-- /explorar/ ações aleatórias para descobrir coisas novas. A probabilidade de
-- explorar é o ε, que começa alto e decai linearmente durante o treino.
module DQN.Policy
  ( epsilonAt,
    greedyAction,
    epsilonGreedy,
  )
where

import DQN.Linalg (argmax)
import DQN.Network (forward)
import DQN.Types
  ( Action,
    Hyperparams (..),
    Network,
    Vector,
    indexToAction,
    numActions,
  )
import System.Random (StdGen, randomR)

-- | ε depois de @t@ passos de treino (passos do agente, não episódios).
--
-- Decai LINEARMENTE de 'hpEpsilonStart' até 'hpEpsilonEnd' ao longo de
-- 'hpEpsilonDecaySteps' passos e depois fica constante em 'hpEpsilonEnd':
--
-- @
-- ε(t) = start + (end − start) · t / decaySteps    se t < decaySteps
-- ε(t) = end                                       se t ≥ decaySteps
-- @
--
-- Se 'hpEpsilonDecaySteps' for 0 (ou negativo), não há decaimento e o
-- resultado é 'hpEpsilonEnd' (evita divisão por zero).
--
-- Pré-condição: @t >= 0@.
epsilonAt :: Hyperparams -> Int -> Double
epsilonAt hp t
  | decaySteps <= 0 || t >= decaySteps = epsEnd
  | otherwise = epsStart + (epsEnd - epsStart) * fraction
  where
    epsStart = hpEpsilonStart hp
    epsEnd = hpEpsilonEnd hp
    decaySteps = hpEpsilonDecaySteps hp
    fraction = fromIntegral t / fromIntegral decaySteps

-- | Ação gulosa: a de maior Q-valor segundo a rede para o estado @s@.
--
-- @a = indexToAction (argmax (Q(s, ·)))@, onde @Q(s, ·) = forward rede s@.
-- Em caso de empate entre os Q-valores, 'argmax' devolve o menor índice.
greedyAction :: Network -> Vector -> Action
greedyAction net s = indexToAction (argmax (forward net s))

-- | Política ε-greedy.
--
-- Sorteia @u@ uniforme em [0, 1):
--
-- * se @u < ε@: EXPLORA, escolhendo uma ação uniformemente entre todas as
--   ações (com um SEGUNDO sorteio, independente de @u@);
-- * senão: APROVEITA, devolvendo 'greedyAction'.
--
-- Como a ação aleatória pode coincidir com a gulosa, com 2 ações a chance de
-- escolher a ação NÃO gulosa é ε / 2.
--
-- O 'StdGen' é passado explicitamente: a função devolve o gerador avançado
-- junto com a ação. Com ε = 0 nunca explora; com ε = 1 explora (quase) sempre.
epsilonGreedy :: Double -> Network -> Vector -> StdGen -> (Action, StdGen)
epsilonGreedy eps net s gen0
  | u < eps = randomAction gen1
  | otherwise = (greedyAction net s, gen1)
  where
    (u, gen1) = randomR (0, 1) gen0
    randomAction g =
      let (i, g') = randomR (0, numActions - 1) g
       in (indexToAction i, g')
