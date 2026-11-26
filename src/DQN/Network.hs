-- |
-- Module      : DQN.Network
-- Description : Parte 3 — a rede neural: inicialização e forward.
--
-- Implementa a função de ativação ReLU, a inicialização de pesos de He
-- e o forward pass (com e sem registro intermediário para o backpropagation).
module DQN.Network
  ( relu,
    reluDerivative,
    initNetwork,
    forwardTrace,
    forward,
  )
where

import DQN.Linalg (matVec, vAdd)
import DQN.Types (Layer (..), LayerTrace (..), Matrix, Network (..), Vector)
import System.Random (StdGen, randomR)

-- ---------------------------------------------------------------------------
-- Função de ativação
-- ---------------------------------------------------------------------------

-- | Função de ativação ReLU (Rectified Linear Unit): @ReLU(z) = max(0, z)@.
relu :: Double -> Double
relu = max 0.0

-- | Derivada da função ReLU: @1@ se @z > 0@, e @0@ caso contrário
-- (inclusive em @z = 0@).
reluDerivative :: Double -> Double
reluDerivative z
  | z > 0.0 = 1.0
  | otherwise = 0.0

-- ---------------------------------------------------------------------------
-- Inicialização da rede
-- ---------------------------------------------------------------------------

-- | Cria uma rede neural densa com os tamanhos de camada dados.
--
-- Por exemplo, @initNetwork [4, 64, 2] gen@ cria 2 camadas: @4 → 64@ e @64 → 2@.
--
-- Os pesos de cada camada com @n_entrada@ entradas são sorteados uniformemente
-- em @[-limite, limite]@, com @limite = sqrt (6 / n_entrada)@ (inicialização
-- de He). Todos os biases começam em @0@.
initNetwork :: [Int] -> StdGen -> (Network, StdGen)
initNetwork sizes gen =
  let pairs = zip sizes (drop 1 sizes)
      (ls, gen') = initLayers pairs gen
   in (Network ls, gen')
  where
    initLayers [] g = ([], g)
    initLayers ((nIn, nOut) : rest) g =
      let (layer, g1) = initLayer nIn nOut g
          (layersRest, g2) = initLayers rest g1
       in (layer : layersRest, g2)

-- | Inicializa uma única camada densa com @nIn@ entradas e @nOut@ saídas.
-- A matriz de pesos tem dimensão @nOut × nIn@ e os biases têm tamanho @nOut@.
initLayer :: Int -> Int -> StdGen -> (Layer, StdGen)
initLayer nIn nOut gen =
  let limit = sqrt (6.0 / fromIntegral nIn)
      (w, gen') = randomMatrix nOut nIn (-limit, limit) gen
      b = replicate nOut 0.0
   in (Layer {weights = w, biases = b}, gen')

-- | Gera uma matriz @rows × cols@ com valores aleatórios uniformes no intervalo dado.
randomMatrix :: Int -> Int -> (Double, Double) -> StdGen -> (Matrix, StdGen)
randomMatrix rows cols range gen
  | rows <= 0 = ([], gen)
  | otherwise =
      let (row, gen1) = randomVector cols range gen
          (rest, gen2) = randomMatrix (rows - 1) cols range gen1
       in (row : rest, gen2)

-- | Gera um vetor de tamanho @n@ com valores aleatórios uniformes no intervalo dado.
randomVector :: Int -> (Double, Double) -> StdGen -> (Vector, StdGen)
randomVector n range gen
  | n <= 0 = ([], gen)
  | otherwise =
      let (x, gen1) = randomR range gen
          (xs, gen2) = randomVector (n - 1) range gen1
       in (x : xs, gen2)

-- ---------------------------------------------------------------------------
-- Forward pass
-- ---------------------------------------------------------------------------

-- | Forward pass guardando os valores intermediários de cada camada, da
-- primeira à última.
--
-- Para cada camada, calcula @z = W · x + b@. As camadas ocultas aplicam ReLU
-- (@a = ReLU(z)@) e a última camada é linear (@traceOutput == tracePreActivation@).
forwardTrace :: Network -> Vector -> [LayerTrace]
forwardTrace (Network ls) = go ls
  where
    go [] _ = []
    go [lastLayer] x =
      let z = vAdd (matVec (weights lastLayer) x) (biases lastLayer)
       in [ LayerTrace
              { traceInput = x,
                tracePreActivation = z,
                traceOutput = z
              }
          ]
    go (layer : rest) x =
      let z = vAdd (matVec (weights layer) x) (biases layer)
          out = map relu z
          trace =
            LayerTrace
              { traceInput = x,
                tracePreActivation = z,
                traceOutput = out
              }
       in trace : go rest out

-- | Calcula a saída da rede (os Q-valores, um por ação), correspondente ao
-- 'traceOutput' da última camada.
forward :: Network -> Vector -> Vector
forward net x = lastOutput x (forwardTrace net x)
  where
    lastOutput fallback [] = fallback
    lastOutput _ [lastTrace] = traceOutput lastTrace
    lastOutput fallback (_ : rest) = lastOutput fallback rest
