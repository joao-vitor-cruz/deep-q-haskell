{-# LANGUAGE BangPatterns #-}

-- |
-- Module      : DQN.Experiment
-- Description : Parte 8 — experimento e resultados.
--
-- Funções puras para exportar registros de treinamento em CSV, calcular
-- médias móveis, avaliar políticas treinadas sem exploração e gerar
-- gráficos de barras em texto para o terminal.
module DQN.Experiment
  ( logsToCsv,
    movingAverage,
    evaluatePolicy,
    asciiChart,
  )
where

import DQN.Env (resetEnv, stateToVector, stepEnv)
import DQN.Policy (greedyAction)
import DQN.Types
  ( CartState,
    EpisodeLog (..),
    Hyperparams (..),
    Network,
    StepResult (..),
  )
import Data.List (intercalate)
import System.Random (StdGen)

-- | Converte a lista de registros de episódios em texto no formato CSV,
-- com o cabeçalho exato @"episode,total_reward,epsilon,avg_loss"@.
logsToCsv :: [EpisodeLog] -> String
logsToCsv logs =
  unlines (header : map formatRow logs)
  where
    header :: String
    header = "episode,total_reward,epsilon,avg_loss"

    formatRow :: EpisodeLog -> String
    formatRow entry =
      intercalate
        ","
        [ show (logEpisode entry),
          show (logTotalReward entry),
          show (logEpsilon entry),
          show (logAvgLoss entry)
        ]

-- | Calcula a média móvel de janela @w@ sobre uma série de valores:
--
-- \[
--   \bar{x}_i = \frac{1}{k} \sum_{j = \max(0,\, i - w + 1)}^{i} x_j
-- \]
--
-- onde \(k = \min(w,\, i + 1)\) é a quantidade de elementos disponíveis até o
-- índice \(i\). Devolve uma lista com o mesmo tamanho da entrada.
movingAverage :: Int -> [Double] -> [Double]
movingAverage window xs
  | window <= 0 = []
  | otherwise = go [] xs
  where
    go :: [Double] -> [Double] -> [Double]
    go _ [] = []
    go acc (y : ys) =
      let acc' = take window (y : acc)
          avg = sum acc' / fromIntegral (length acc')
       in avg : go acc' ys

-- | Executa @n@ episódios completos usando apenas a política gulosa
-- (\(\varepsilon = 0\), sem atualização de pesos) e devolve a recompensa total
-- obtida em cada episódio, respeitando 'hpMaxStepsPerEpisode'.
evaluatePolicy :: Hyperparams -> Network -> Int -> StdGen -> ([Double], StdGen)
evaluatePolicy hp net episodes gen
  | episodes <= 0 = ([], gen)
  | otherwise = go episodes gen []
  where
    maxSteps :: Int
    maxSteps = hpMaxStepsPerEpisode hp

    go :: Int -> StdGen -> [Double] -> ([Double], StdGen)
    go 0 g acc = (reverse acc, g)
    go k g acc =
      let (s0, g') = resetEnv g
          !epReward = runGreedyEpisode s0 0 0.0
       in go (k - 1) g' (epReward : acc)

    runGreedyEpisode :: CartState -> Int -> Double -> Double
    runGreedyEpisode !state !step !totalR
      | step >= maxSteps = totalR
      | otherwise =
          let action = greedyAction net (stateToVector state)
              res = stepEnv state action
              !totalR' = totalR + reward res
           in if done res
                then totalR'
                else runGreedyEpisode (nextState res) (step + 1) totalR'

-- | Gera um gráfico de barras vertical em ASCII de dimensões @width@ colunas
-- por @height@ linhas. Se houver mais pontos do que @width@, agrupa os dados
-- por média em blocos para caber na largura especificada.
asciiChart :: Int -> Int -> [Double] -> String
asciiChart width height values
  | width <= 0 || height <= 0 = ""
  | null values = unlines (replicate height (replicate width ' '))
  | otherwise = unlines [renderRow row | row <- [height, height - 1 .. 1]]
  where
    binned :: [Double]
    binned = resample width values

    minVal :: Double
    minVal = minimum binned

    maxVal :: Double
    maxVal = maximum binned

    spanVal :: Double
    spanVal = maxVal - minVal

    barHeights :: [Int]
    barHeights = map toBarHeight binned

    toBarHeight :: Double -> Int
    toBarHeight v
      | spanVal <= 1e-12 = if maxVal > 0 then height else 0
      | otherwise =
          let norm = (v - minVal) / spanVal
              scaled = round (norm * fromIntegral height)
           in max 0 (min height scaled)

    renderRow :: Int -> String
    renderRow r = [if h >= r then '#' else ' ' | h <- barHeights]

-- | Redimensiona uma lista não vazia para ter exatamente @targetWidth@ pontos,
-- calculando a média por intervalo quando @length xs > targetWidth@.
resample :: Int -> [Double] -> [Double]
resample targetWidth xs
  | n <= targetWidth = xs
  | otherwise = [bucketAvg i | i <- [0 .. targetWidth - 1]]
  where
    n :: Int
    n = length xs

    bucketAvg :: Int -> Double
    bucketAvg i =
      let startIdx = (i * n) `div` targetWidth
          endIdx = max (startIdx + 1) (((i + 1) * n) `div` targetWidth)
          slice = take (endIdx - startIdx) (drop startIdx xs)
       in sum slice / fromIntegral (length slice)