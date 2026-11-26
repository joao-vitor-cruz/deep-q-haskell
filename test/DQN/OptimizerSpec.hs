module DQN.OptimizerSpec (spec) where

import DQN.Optimizer (adamStep, initAdam, sgdStep)
import DQN.Types
  ( AdamState (..),
    Layer (..),
    Network (..),
  )
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)

sampleNetwork :: Network
sampleNetwork =
  Network
    [ Layer
        { weights = [[1.0, -2.0], [0.5, 3.0]],
          biases = [0.25, -0.5]
        }
    ]

sampleGrad :: Network
sampleGrad =
  Network
    [ Layer
        { weights = [[2.0, -4.0], [-1.0, 6.0]],
          biases = [1.0, -2.0]
        }
    ]

zeroGrad :: Network
zeroGrad =
  Network
    [ Layer
        { weights = [[0.0, 0.0], [0.0, 0.0]],
          biases = [0.0, 0.0]
        }
    ]

firstWeight :: Network -> Double
firstWeight (Network (Layer {weights = ((w : _) : _)} : _)) = w
firstWeight _ = error "Rede vazia em firstWeight"

scalarNetwork :: Double -> Network
scalarNetwork w =
  Network
    [ Layer
        { weights = [[w]],
          biases = [0.0]
        }
    ]

scalarGrad :: Double -> Network
scalarGrad w =
  Network
    [ Layer
        { weights = [[2.0 * w]],
          biases = [0.0]
        }
    ]

runScalarAdam :: Int -> Double -> Double -> (Network, AdamState)
runScalarAdam 0 w0 _ = (scalarNetwork w0, initAdam (scalarNetwork w0))
runScalarAdam n w0 lr = go n initNet initSt
  where
    initNet = scalarNetwork w0
    initSt = initAdam initNet
    go 0 net st = (net, st)
    go k net st =
      let g = scalarGrad (firstWeight net)
          (net', st') = adamStep lr st net g
       in go (k - 1) net' st'

spec :: Spec
spec = do
  describe "DQN.Optimizer" $ do
    it "initAdam cria momentos zerados e contador 0" $ do
      let st = initAdam sampleNetwork
      adamStepCount st `shouldBe` 0
      adamM st `shouldBe` zeroGrad
      adamV st `shouldBe` zeroGrad

    it "adamStep incrementa o contador em 1 a cada passo" $ do
      let st0 = initAdam sampleNetwork
          (_, st1) = adamStep 0.01 st0 sampleNetwork sampleGrad
          (_, st2) = adamStep 0.01 st1 sampleNetwork sampleGrad
      adamStepCount st1 `shouldBe` 1
      adamStepCount st2 `shouldBe` 2

    it "no primeiro passo do Adam, o peso anda aprox. lr na direcao oposta ao gradiente" $ do
      let lr = 0.01
          st0 = initAdam sampleNetwork
          (net1, _) = adamStep lr st0 sampleNetwork sampleGrad
          wBefore = firstWeight sampleNetwork
          wAfter = firstWeight net1
      wAfter `shouldSatisfy` (< wBefore)
      abs ((wBefore - wAfter) - lr) `shouldSatisfy` (< 1e-4)

    it "sgdStep com gradiente zero nao altera a rede" $ do
      sgdStep 0.1 sampleNetwork zeroGrad `shouldBe` sampleNetwork

    it "sgdStep aplica w' = w - lr * g corretamente" $ do
      let expected =
            Network
              [ Layer
                  { weights = [[0.8, -1.6], [0.6, 2.4]],
                    biases = [0.15, -0.3]
                  }
              ]
      sgdStep 0.1 sampleNetwork sampleGrad `shouldBe` expected

    it "aplicar adamStep repetidamente em f(w) = w^2 leva w para perto de 0" $ do
      let (finalNet, _) = runScalarAdam 500 5.0 0.05
          wFinal = firstWeight finalNet
      abs wFinal `shouldSatisfy` (< 1e-2)