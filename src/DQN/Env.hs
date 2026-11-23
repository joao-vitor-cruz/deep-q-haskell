-- |
-- Module      : DQN.Env
-- Description : Parte 1 — o ambiente CartPole.
--
-- Simula a dinâmica física do problema clássico CartPole (Barto, Sutton e
-- Anderson, 1983), equivalente ao ambiente CartPole-v1 do Gymnasium.
module DQN.Env
  ( resetEnv,
    stepEnv,
    isTerminal,
    stateToVector,
  )
where

import DQN.Types
  ( Action (..),
    CartState (..),
    StepResult (..),
    Vector,
  )
import System.Random (StdGen, randomR)

-- ---------------------------------------------------------------------------
-- Constantes físicas do CartPole-v1
-- ---------------------------------------------------------------------------

-- | Aceleração da gravidade (m/s²).
gravity :: Double
gravity = 9.8

-- | Massa do carrinho (kg).
massCart :: Double
massCart = 1.0

-- | Massa da haste (kg).
massPole :: Double
massPole = 0.1

-- | Massa total do sistema: M = m_c + m_p (kg).
totalMass :: Double
totalMass = massCart + massPole

-- | Metade do comprimento da haste (m).
halfPoleLength :: Double
halfPoleLength = 0.5

-- | Produto da massa da haste pela sua metade de comprimento: m_p * l (kg*m).
poleMassLength :: Double
poleMassLength = massPole * halfPoleLength

-- | Magnitude da força horizontal aplicada ao carrinho (N).
forceMag :: Double
forceMag = 10.0

-- | Intervalo de tempo de cada passo da simulação (s).
tau :: Double
tau = 0.02

-- | Limite máximo de deslocamento horizontal do carrinho a partir do centro (m).
xThreshold :: Double
xThreshold = 2.4

-- | Limite máximo de inclinação da haste em radianos: 12 graus = 12 * pi / 180.
thetaThresholdRadians :: Double
thetaThresholdRadians = 12.0 * pi / 180.0

-- ---------------------------------------------------------------------------
-- Funções exportadas
-- ---------------------------------------------------------------------------

-- | Gera o estado inicial do episódio sorteando cada um dos 4 componentes
-- uniformemente no intervalo [-0.05, 0.05] e devolvendo o novo gerador.
resetEnv :: StdGen -> (CartState, StdGen)
resetEnv gen0 =
  let (x, gen1) = randomR (-0.05, 0.05) gen0
      (xDot, gen2) = randomR (-0.05, 0.05) gen1
      (theta, gen3) = randomR (-0.05, 0.05) gen2
      (thetaDot, gen4) = randomR (-0.05, 0.05) gen3
   in ( CartState
          { cartPosition = x,
            cartVelocity = xDot,
            poleAngle = theta,
            poleAngularVelocity = thetaDot
          },
        gen4
      )

-- | Aplica uma ação ('PushLeft' com F = -10 N ou 'PushRight' com F = +10 N)
-- por um passo de tempo tau = 0.02 s.
--
-- Equações do movimento:
--
-- * temp = (F + m_p * l * thetaDot^2 * sin(theta)) / M
-- * thetaAcc = (g * sin(theta) - cos(theta) * temp) / (l * (4/3 - m_p * cos^2(theta) / M))
-- * xAcc = temp - m_p * l * thetaAcc * cos(theta) / M
--
-- Integração de Euler explícita (usando os valores anteriores do estado):
--
-- * x' = x + tau * xDot
-- * xDot' = xDot + tau * xAcc
-- * theta' = theta + tau * thetaDot
-- * thetaDot' = thetaDot + tau * thetaAcc
--
-- A recompensa é sempre 1.0, inclusive no passo em que o episódio termina.
stepEnv :: CartState -> Action -> StepResult
stepEnv state action =
  let x = cartPosition state
      xDot = cartVelocity state
      theta = poleAngle state
      thetaDot = poleAngularVelocity state

      force = case action of
        PushRight -> forceMag
        PushLeft -> -forceMag

      cosTheta = cos theta
      sinTheta = sin theta

      temp = (force + poleMassLength * thetaDot * thetaDot * sinTheta) / totalMass
      denom = halfPoleLength * (4.0 / 3.0 - massPole * cosTheta * cosTheta / totalMass)
      thetaAcc = (gravity * sinTheta - cosTheta * temp) / denom
      xAcc = temp - poleMassLength * thetaAcc * cosTheta / totalMass

      x' = x + tau * xDot
      xDot' = xDot + tau * xAcc
      theta' = theta + tau * thetaDot
      thetaDot' = thetaDot + tau * thetaAcc

      next =
        CartState
          { cartPosition = x',
            cartVelocity = xDot',
            poleAngle = theta',
            poleAngularVelocity = thetaDot'
          }
   in StepResult
        { nextState = next,
          reward = 1.0,
          done = isTerminal next
        }

-- | Verifica se o episódio terminou.
-- Retorna 'True' quando |posição| > 2.4 m ou |ângulo| > 12 graus (12 * pi / 180 rad).
isTerminal :: CartState -> Bool
isTerminal state =
  abs (cartPosition state) > xThreshold
    || abs (poleAngle state) > thetaThresholdRadians

-- | Converte um 'CartState' no vetor de entrada da rede neural, na ordem exata:
-- @[posição, velocidade, ângulo, velocidade angular]@.
stateToVector :: CartState -> Vector
stateToVector state =
  [ cartPosition state,
    cartVelocity state,
    poleAngle state,
    poleAngularVelocity state
  ]