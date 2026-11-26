module DQN.PolicySpec (spec) where

import DQN.Policy (epsilonAt, epsilonGreedy, greedyAction)
import DQN.Types
  ( Action (..),
    Hyperparams (..),
    Layer (..),
    Network (..),
    Vector,
    defaultHyperparams,
  )
import System.Random (StdGen, mkStdGen)
import Test.Hspec

-- | Hiperparâmetros com os valores de exploração fixados aqui, para o teste
-- não quebrar se alguém mudar o 'defaultHyperparams'.
hp :: Hyperparams
hp =
  defaultHyperparams
    { hpEpsilonStart = 1.0,
      hpEpsilonEnd = 0.05,
      hpEpsilonDecaySteps = 10000
    }

-- | Rede de uma camada montada à mão: Q = W x, com W = [[1,0,0,0],[0,1,0,0]]
-- e bias zero. Logo Q(PushLeft) = x0 e Q(PushRight) = x1.
handNet :: Network
handNet =
  Network
    [ Layer
        { weights = [[1, 0, 0, 0], [0, 1, 0, 0]],
          biases = [0, 0]
        }
    ]

-- | Entrada em que Q = [1, 0]: a ação gulosa é PushLeft.
favorLeft :: Vector
favorLeft = [1, 0, 0, 0]

-- | Entrada em que Q = [0, 1]: a ação gulosa é PushRight.
favorRight :: Vector
favorRight = [0, 1, 0, 0]

-- | Compara números reais com tolerância (evita erro de arredondamento).
closeTo :: Double -> Double -> Bool
closeTo expected actual = abs (actual - expected) < 1e-9

-- | Chama 'epsilonGreedy' @n@ vezes, passando o gerador adiante a cada sorteio.
sampleActions :: Int -> Double -> Network -> Vector -> StdGen -> [Action]
sampleActions n eps net s gen
  | n <= 0 = []
  | otherwise =
      let (a, gen') = epsilonGreedy eps net s gen
       in a : sampleActions (n - 1) eps net s gen'

-- | Testes da Parte 6 (exploração ε-greedy).
spec :: Spec
spec =
  describe "DQN.Policy" $ do
    describe "epsilonAt" $ do
      it "no passo 0 devolve hpEpsilonStart" $
        epsilonAt hp 0 `shouldBe` hpEpsilonStart hp

      it "em hpEpsilonDecaySteps devolve hpEpsilonEnd" $
        epsilonAt hp (hpEpsilonDecaySteps hp) `shouldBe` hpEpsilonEnd hp

      it "depois do decaimento continua constante em hpEpsilonEnd" $ do
        let n = hpEpsilonDecaySteps hp
        map (epsilonAt hp) [n + 1, 2 * n, 100 * n]
          `shouldBe` replicate 3 (hpEpsilonEnd hp)

      it "no meio do caminho devolve o valor intermediario" $
        epsilonAt hp 5000 `shouldSatisfy` closeTo 0.525

      it "com hpEpsilonDecaySteps = 0 devolve hpEpsilonEnd sem dividir por zero" $
        epsilonAt hp {hpEpsilonDecaySteps = 0} 0 `shouldBe` hpEpsilonEnd hp

    describe "greedyAction" $ do
      it "escolhe PushLeft quando Q(PushLeft) e o maior" $
        greedyAction handNet favorLeft `shouldBe` PushLeft

      it "escolhe PushRight quando Q(PushRight) e o maior" $
        greedyAction handNet favorRight `shouldBe` PushRight

    describe "epsilonGreedy" $ do
      it "com epsilon 0 devolve sempre a acao gulosa" $ do
        let gens = map mkStdGen [1 .. 50]
            acoes s = [fst (epsilonGreedy 0 handNet s g) | g <- gens]
        all (== greedyAction handNet favorLeft) (acoes favorLeft) `shouldBe` True
        all (== greedyAction handNet favorRight) (acoes favorRight) `shouldBe` True

      it "com epsilon 1 as duas acoes aparecem em 200 sorteios" $ do
        let acoes = sampleActions 200 1 handNet favorRight (mkStdGen 42)
        acoes `shouldSatisfy` elem PushLeft
        acoes `shouldSatisfy` elem PushRight
