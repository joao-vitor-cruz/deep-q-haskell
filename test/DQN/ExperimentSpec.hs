module DQN.ExperimentSpec (spec) where

import DQN.Experiment
  ( asciiChart,
    evaluatePolicy,
    logsToCsv,
    movingAverage,
  )
import DQN.Network (initNetwork)
import DQN.Types
  ( EpisodeLog (..),
    Hyperparams (..),
    defaultHyperparams,
  )
import System.Random (mkStdGen)
import Test.Hspec

-- | Testes da Parte 8 (experimento e resultados).
spec :: Spec
spec =
  describe "DQN.Experiment" $ do
    it "gera CSV com cabecalho exato e uma linha por episodio" $ do
      let sampleLogs =
            [ EpisodeLog 1 15.0 0.99 0.12,
              EpisodeLog 2 22.0 0.98 0.08
            ]
          csvLines = lines (logsToCsv sampleLogs)
      length csvLines `shouldBe` 3
      head csvLines `shouldBe` "episode,total_reward,epsilon,avg_loss"
      csvLines !! 1 `shouldBe` "1,15.0,0.99,0.12"
      csvLines !! 2 `shouldBe` "2,22.0,0.98,0.08"

    it "calcula corretamente a media movel com janela parcial no inicio" $ do
      movingAverage 2 [2, 4, 6] `shouldBe` [2.0, 3.0, 5.0]
      movingAverage 3 [10, 20, 30, 40] `shouldBe` [10.0, 15.0, 20.0, 30.0]

    it "avalia a politica gulosa devolvendo n recompensas dentro do limite" $ do
      let hp = defaultHyperparams {hpMaxStepsPerEpisode = 50}
          (net, gen1) = initNetwork (hpLayerSizes hp) (mkStdGen 42)
          (rewards, _) = evaluatePolicy hp net 10 gen1
      length rewards `shouldBe` 10
      all (\r -> r >= 1.0 && r <= 50.0) rewards `shouldBe` True

    it "gera grafico ASCII com a quantidade exata de linhas solicitada" $ do
      let chart = asciiChart 10 5 [10.0, 25.0, 50.0, 100.0, 200.0]
      length (lines chart) `shouldBe` 5