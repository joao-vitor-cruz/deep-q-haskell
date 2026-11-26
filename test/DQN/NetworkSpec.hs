module DQN.NetworkSpec (spec) where

import DQN.Network (forward, forwardTrace, initNetwork, relu, reluDerivative)
import DQN.Types (Layer (..), LayerTrace (..), Network (..))
import System.Random (mkStdGen)
import Test.Hspec

-- | Compara dois valores Double dentro de uma tolerancia pequena.
shouldBeApprox :: Double -> Double -> Expectation
shouldBeApprox actual expected =
  abs (actual - expected) `shouldSatisfy` (< 1e-9)

-- | Testes da Parte 3 (a rede neural: inicializacao e forward).
spec :: Spec
spec =
  describe "DQN.Network" $ do
    describe "relu e reluDerivative" $ do
      it "relu devolve 0 para entradas negativas e zero, e o proprio valor para positivas" $ do
        relu (-3.5) `shouldBe` 0.0
        relu 0.0 `shouldBe` 0.0
        relu 2.75 `shouldBe` 2.75

      it "reluDerivative devolve 1 apenas para z > 0 e 0 para z <= 0" $ do
        reluDerivative (-1.0) `shouldBe` 0.0
        reluDerivative 0.0 `shouldBe` 0.0
        reluDerivative 0.001 `shouldBe` 1.0

    describe "initNetwork" $ do
      let gen0 = mkStdGen 42
          (net, gen1) = initNetwork [4, 64, 2] gen0

      it "gera 2 camadas com dimensoes de pesos 64x4 e 2x64 e biases 64 e 2" $ do
        length (layers net) `shouldBe` 2
        case layers net of
          [l1, l2] -> do
            map length (weights l1) `shouldBe` replicate 64 4
            length (biases l1) `shouldBe` 64
            map length (weights l2) `shouldBe` replicate 2 64
            length (biases l2) `shouldBe` 2
          _ -> expectationFailure "esperava exatamente 2 camadas"

      it "inicializa pesos dentro do limite de He e biases zerados" $ do
        let limit1 = sqrt (6.0 / 4.0)
            limit2 = sqrt (6.0 / 64.0)
            inBounds lim w = w >= -lim && w <= lim
        case layers net of
          [l1, l2] -> do
            all (all (inBounds limit1)) (weights l1) `shouldBe` True
            all (all (inBounds limit2)) (weights l2) `shouldBe` True
            biases l1 `shouldBe` replicate 64 0.0
            biases l2 `shouldBe` replicate 2 0.0
          _ -> expectationFailure "esperava exatamente 2 camadas"

      it "nao gera todos os pesos iguais e avanca o StdGen" $ do
        let allWeights = [w | layer <- layers net, row <- weights layer, w <- row]
        case allWeights of
          (w0 : ws) -> any (/= w0) ws `shouldBe` True
          [] -> expectationFailure "rede sem pesos"
        show gen1 `shouldNotBe` show gen0

    describe "forward e forwardTrace" $ do
      it "forward devolve vetor de tamanho 2 para rede [4, 64, 2]" $ do
        let (net, _) = initNetwork [4, 64, 2] (mkStdGen 7)
            out = forward net [0.1, -0.2, 0.05, 0.3]
        length out `shouldBe` 2

      it "calcula exatamente a saida esperada em uma rede montada a mao" $ do
        -- Camada oculta (2 -> 2, ReLU):
        --   W1 = [[1, -2], [-1, 3]], b1 = [0.5, -1.0]
        --   x = [2, 1]
        --   z1 = [1*2 + (-2)*1 + 0.5, (-1)*2 + 3*1 - 1.0] = [0.5, 0.0]
        --   a1 = ReLU(z1) = [0.5, 0.0]
        -- Camada de saida (2 -> 2, linear):
        --   W2 = [[2, 4], [-3, 1]], b2 = [1.0, -0.5]
        --   z2 = [2*0.5 + 4*0.0 + 1.0, (-3)*0.5 + 1*0.0 - 0.5] = [2.0, -2.0]
        let l1 = Layer {weights = [[1.0, -2.0], [-1.0, 3.0]], biases = [0.5, -1.0]}
            l2 = Layer {weights = [[2.0, 4.0], [-3.0, 1.0]], biases = [1.0, -0.5]}
            net = Network [l1, l2]
            out = forward net [2.0, 1.0]
        length out `shouldBe` 2
        case out of
          [q0, q1] -> do
            q0 `shouldBeApprox` 2.0
            q1 `shouldBeApprox` (-2.0)
          _ -> expectationFailure "esperava saida de tamanho 2"

      it "permite saidas negativas na ultima camada (ativacao linear)" $ do
        let l1 = Layer {weights = [[1.0, 1.0]], biases = [0.0]}
            l2 = Layer {weights = [[-2.0], [-5.0]], biases = [-1.0, -0.5]}
            net = Network [l1, l2]
            out = forward net [1.0, 2.0]
        length out `shouldBe` 2
        all (< 0.0) out `shouldBe` True

      it "forwardTrace tem um trace por camada, ReLU nas ocultas e ultima camada linear" $ do
        let (net, _) = initNetwork [4, 5, 3, 2] (mkStdGen 123)
            x = [0.2, -0.1, 0.03, -0.4]
            traces = forwardTrace net x
        length traces `shouldBe` length (layers net)
        case traces of
          [t1, t2, t3] -> do
            traceInput t1 `shouldBe` x
            traceOutput t1 `shouldBe` map relu (tracePreActivation t1)
            traceInput t2 `shouldBe` traceOutput t1
            traceOutput t2 `shouldBe` map relu (tracePreActivation t2)
            traceInput t3 `shouldBe` traceOutput t2
            traceOutput t3 `shouldBe` tracePreActivation t3
            traceOutput t3 `shouldBe` forward net x
          _ -> expectationFailure "esperava 3 traces"
