-- |
-- Module      : DQN.Backprop
-- Description : Parte 4 — perda e gradientes.
--
-- Calcula a perda (Huber) e os gradientes da rede em relação a todos os
-- pesos e biases, via backpropagation. Este módulo só CALCULA o gradiente;
-- quem atualiza os pesos é o "DQN.Optimizer" (Parte 9).
module DQN.Backprop
  ( huberLoss,
    huberLossDerivative,
    backprop,
    sampleGradient,
    batchGradient,
  )
where

import DQN.Linalg (addNetworks, hadamard, matVec, outer, scaleNetwork, transposeM)
import DQN.Network (forward, forwardTrace, reluDerivative)
import DQN.Types (Layer (..), LayerTrace (..), Network (..), Vector)
import Data.List (foldl')

-- ---------------------------------------------------------------------------
-- Perda de Huber
-- ---------------------------------------------------------------------------

-- | Perda de Huber com δ = 1, recebendo (previsão, alvo).
--
-- Com d = previsão − alvo:
--
-- * 0.5 · d²     se |d| ≤ 1
-- * |d| − 0.5    caso contrário
--
-- Perto de zero é igual ao erro quadrático; longe de zero cresce só
-- linearmente, o que evita gradientes enormes quando o alvo está muito errado.
huberLoss :: Double -> Double -> Double
huberLoss prediction target
  | abs d <= 1 = 0.5 * d * d
  | otherwise = abs d - 0.5
  where
    d = prediction - target

-- | Derivada da perda de Huber em relação à PREVISÃO.
--
-- dL/dprevisão = clamp(d, −1, 1), com d = previsão − alvo.
-- (No trecho quadrático a derivada é d; no trecho linear é o sinal de d.)
huberLossDerivative :: Double -> Double -> Double
huberLossDerivative prediction target = max (-1) (min 1 (prediction - target))

-- ---------------------------------------------------------------------------
-- Backpropagation
-- ---------------------------------------------------------------------------

-- | Backpropagation. Recebe a rede, a entrada x e o gradiente da perda em
-- relação à SAÍDA da rede (dL/dsaída). Devolve o gradiente da perda em
-- relação a todos os pesos e biases, no formato de uma 'Network' (mesmas
-- dimensões da rede, camadas na mesma ordem: da primeira para a última).
--
-- Percorre as camadas da ÚLTIMA para a PRIMEIRA. Com dA = gradiente em
-- relação à saída da camada (começando com dL/dsaída):
--
-- > dz     = dA                       (última camada, linear)
-- > dz     = dA ⊙ ReLU'(z)            (camadas ocultas)
-- > dW     = outer dz entrada         (mesmo formato de W)
-- > db     = dz
-- > dA_ant = Wᵀ · dz                  (vira o dA da camada anterior)
backprop :: Network -> Vector -> Vector -> Network
backprop net x dOutput = Network (reverse gradsLastToFirst)
  where
    -- Cada camada junto com o que ela guardou no forward (entrada e z),
    -- em ordem invertida: da última camada para a primeira.
    layersLastToFirst = reverse (zip (layers net) (forwardTrace net x))
    gradsLastToFirst = go True dOutput layersLastToFirst

    -- isLast indica se é a última camada (a única sem ReLU).
    go :: Bool -> Vector -> [(Layer, LayerTrace)] -> [Layer]
    go _ _ [] = []
    go isLast dA ((layer, trace) : rest) = grad : go False dAPrev rest
      where
        dz
          | isLast = dA
          | otherwise = hadamard dA (map reluDerivative (tracePreActivation trace))
        grad = Layer {weights = outer dz (traceInput trace), biases = dz}
        dAPrev = matVec (transposeM (weights layer)) dz

-- ---------------------------------------------------------------------------
-- Gradiente de uma amostra e de um minibatch
-- ---------------------------------------------------------------------------

-- | Gradiente e perda para UMA amostra: entrada x, índice da ação a, alvo y.
--
-- A perda só olha a saída da ação a: L = huberLoss (Q(x)[a]) y.
-- O gradiente em relação à saída é zero em todas as posições, exceto na
-- posição a, onde vale huberLossDerivative (Q(x)[a]) y.
--
-- Pré-condição: 0 ≤ a < número de saídas da rede.
sampleGradient :: Network -> Vector -> Int -> Double -> (Network, Double)
sampleGradient net x a y = (backprop net x dOutput, huberLoss qa y)
  where
    q = forward net x
    qa = q !! a
    dOutput = [if i == a then huberLossDerivative qa y else 0 | i <- [0 .. length q - 1]]

-- | Média dos gradientes e média das perdas de um minibatch de (x, a, y):
--
-- > gradiente = (1/N) · Σ gradienteᵢ        perda = (1/N) · Σ perdaᵢ
--
-- Pré-condição: lista não vazia.
batchGradient :: Network -> [(Vector, Int, Double)] -> (Network, Double)
batchGradient _ [] = error "DQN.Backprop.batchGradient: minibatch vazio"
batchGradient net samples@(first : rest) =
  (scaleNetwork invN gradSum, lossSum * invN)
  where
    gradientOf (x, a, y) = sampleGradient net x a y
    (gradSum, lossSum) = foldl' accumulate (gradientOf first) rest
    accumulate (accGrad, accLoss) sample =
      let (g, l) = gradientOf sample
       in (addNetworks accGrad g, accLoss + l)
    invN = 1 / fromIntegral (length samples)
