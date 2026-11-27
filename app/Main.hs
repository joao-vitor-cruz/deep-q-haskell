{-# LANGUAGE BangPatterns #-}

-- |
-- Module      : Main
-- Description : Ponto de entrada do experimento DQN no CartPole (Parte 8).
--
-- Único módulo do projeto que realiza operações de entrada e saída (IO).
module Main (main) where

import DQN.Agent (train)
import DQN.Env (resetEnv, stepEnv)
import DQN.Experiment
  ( asciiChart,
    evaluatePolicy,
    logsToCsv,
    movingAverage,
  )
import DQN.Types
  ( Agent (..),
    CartState,
    EpisodeLog (..),
    Hyperparams (..),
    StepResult (..),
    defaultHyperparams,
    indexToAction,
    numActions,
  )
import System.Directory (createDirectoryIfMissing)
import System.Environment (getArgs)
import System.Random (StdGen, mkStdGen, uniformR)
import Text.Printf (printf)
import Text.Read (readMaybe)
import Data.Maybe (fromMaybe)

main :: IO ()
main = do
  args <- getArgs
  let (seed, episodes) = parseArgs args
      hp = defaultHyperparams {hpEpisodes = episodes}
      gen0 = mkStdGen seed

  putStrLn "=== Deep Q-Learning (DQN) em Haskell puro - CartPole ==="
  printf "Semente: %d | Episodios de treino: %d\n" seed episodes
  putStrLn "Treinando agente (isso pode levar alguns minutos)..."

  let (trainedAgent, logs) = train hp gen0
      csvPath = "results/training.csv"

  createDirectoryIfMissing True "results"
  writeFile csvPath (logsToCsv logs)
  printf "Registros de treinamento salvos em: %s\n\n" csvPath

  let rewards = map logTotalReward logs
      ma20 = movingAverage 20 rewards
      chartWidth = min 60 (max 1 (length ma20))
      chartHeight = 12

  putStrLn "Curva de aprendizado (media movel de 20 episodios):"
  putStr (asciiChart chartWidth chartHeight ma20)
  putStrLn (replicate chartWidth '-')
  if null ma20
    then putStrLn "Sem dados de episodios para exibir."
    else
      printf
        "Min media movel: %.2f | Max media movel: %.2f\n\n"
        (minimum ma20)
        (maximum ma20)

  let evalEpisodes = 20
      genEval0 = mkStdGen (seed + 1000)
      (randRewards, genEval1) = evaluateRandomPolicy hp evalEpisodes genEval0
      (finalRewards, genEval2) = evaluatePolicy hp (agentOnline trainedAgent) evalEpisodes genEval1
      (bestRewards, _) = evaluatePolicy hp (agentBest trainedAgent) evalEpisodes genEval2

  putStrLn "Avaliacao comparativa (20 episodios cada, sem exploracao):"
  printf "  1. Politica aleatoria : %.2f passos em media\n" (mean randRewards)
  printf "  2. Rede final (online): %.2f passos em media\n" (mean finalRewards)
  printf "  3. Melhor rede (best) : %.2f passos em media\n" (mean bestRewards)

-- | Interpreta argumentos opcionais da linha de comando: [semente] [episodios].
parseArgs :: [String] -> (Int, Int)
parseArgs args =
  let defaultSeed = 42
      defaultEps = hpEpisodes defaultHyperparams
      seed = case args of
        (s : _) -> fromMaybe defaultSeed (readMaybe s)
        [] -> defaultSeed
      eps = case args of
        (_ : e : _) -> fromMaybe defaultEps (readMaybe e)
        _ -> defaultEps
   in (seed, max 1 eps)

-- | Avalia uma política puramente aleatória (baseline) por @n@ episódios.
evaluateRandomPolicy :: Hyperparams -> Int -> StdGen -> ([Double], StdGen)
evaluateRandomPolicy hp episodes gen = go episodes gen []
  where
    maxSteps :: Int
    maxSteps = hpMaxStepsPerEpisode hp

    go :: Int -> StdGen -> [Double] -> ([Double], StdGen)
    go 0 g acc = (reverse acc, g)
    go k g acc =
      let (s0, g1) = resetEnv g
          (!epR, g2) = runRandomEpisode s0 0 0.0 g1
       in go (k - 1) g2 (epR : acc)

    runRandomEpisode :: CartState -> Int -> Double -> StdGen -> (Double, StdGen)
    runRandomEpisode !state !step !totalR g
      | step >= maxSteps = (totalR, g)
      | otherwise =
          let (idx, g') = uniformR (0, numActions - 1) g
              res = stepEnv state (indexToAction idx)
              !totalR' = totalR + reward res
           in if done res
                then (totalR', g')
                else runRandomEpisode (nextState res) (step + 1) totalR' g'

-- | Média aritmética simples de uma lista de valores.
mean :: [Double] -> Double
mean [] = 0.0
mean xs = sum xs / fromIntegral (length xs)