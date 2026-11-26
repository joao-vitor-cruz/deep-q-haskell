-- |
-- Module      : DQN.Optimizer
-- Description : Otimizadores SGD e Adam para atualização dos pesos da rede neural.
module DQN.Optimizer
  ( initAdam,
    sgdStep,
    adamStep,
  )
where

import DQN.Linalg (zeroNetwork, zipNetworkWith)
import DQN.Types (AdamState (..), Network)

-- | Estado inicial do Adam para uma rede: momentos zerados ('zeroNetwork')
-- e contador de passos em 0.
initAdam :: Network -> AdamState
initAdam net =
  AdamState
    { adamM = zeroNetwork net,
      adamV = zeroNetwork net,
      adamStepCount = 0
    }

-- | Gradiente descendente simples (SGD):
--
-- @w' = w - lr * g@
--
-- Recebe a taxa de aprendizado (@lr@), a rede atual e o gradiente, devolvendo
-- a rede atualizada.
sgdStep :: Double -> Network -> Network -> Network
sgdStep lr = zipNetworkWith (\w g -> w - lr * g)

-- | Um passo do otimizador Adam (Kingma e Ba, 2014), com @beta1 = 0.9@,
-- @beta2 = 0.999@ e @eps = 1e-8@:
--
-- @
-- t' = t + 1
-- m' = beta1 * m + (1 - beta1) * g
-- v' = beta2 * v + (1 - beta2) * g^2
-- m_hat = m' / (1 - beta1 ^ t')
-- v_hat = v' / (1 - beta2 ^ t')
-- w' = w - lr * m_hat / (sqrt v_hat + eps)
-- @
--
-- Recebe a taxa de aprendizado, o estado atual do Adam, a rede atual e o
-- gradiente; devolve @(redeNova, estadoNovo)@.
adamStep :: Double -> AdamState -> Network -> Network -> (Network, AdamState)
adamStep lr st net grad = (newNet, newState)
  where
    beta1 :: Double
    beta1 = 0.9

    beta2 :: Double
    beta2 = 0.999

    eps :: Double
    eps = 1e-8

    t' :: Int
    t' = adamStepCount st + 1

    biasCorr1 :: Double
    biasCorr1 = 1.0 - beta1 ^ t'

    biasCorr2 :: Double
    biasCorr2 = 1.0 - beta2 ^ t'

    m' :: Network
    m' = zipNetworkWith (\m g -> beta1 * m + (1.0 - beta1) * g) (adamM st) grad

    v' :: Network
    v' = zipNetworkWith (\v g -> beta2 * v + (1.0 - beta2) * g * g) (adamV st) grad

    stepNet :: Network
    stepNet =
      zipNetworkWith
        ( \mVal vVal ->
            let mHat = mVal / biasCorr1
                vHat = vVal / biasCorr2
             in lr * mHat / (sqrt vHat + eps)
        )
        m'
        v'

    newNet :: Network
    newNet = zipNetworkWith (-) net stepNet

    newState :: AdamState
    newState =
      AdamState
        { adamM = m',
          adamV = v',
          adamStepCount = t'
        }