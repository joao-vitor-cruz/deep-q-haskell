module DQN.LinalgSpec (spec) where

import DQN.Linalg
import DQN.Types (Layer (..), Network (..))
import Test.Hspec

-- | Rede pequena de exemplo usada nos testes de operações sobre Network.
sampleNet :: Network
sampleNet =
  Network
    [ Layer {weights = [[1.0, 2.0], [3.0, 4.0]], biases = [0.5, -0.5]},
      Layer {weights = [[2.0, -1.0]], biases = [1.0]}
    ]

spec :: Spec
spec = describe "DQN.Linalg" $ do
  describe "operacoes com vetores" $ do
    it "calcula dot corretamente" $
      dot [1, 2, 3] [4, 5, 6] `shouldBe` 32

    it "soma vetores com vAdd" $
      vAdd [1, 2, 3] [4, 5, 6] `shouldBe` [5, 7, 9]

    it "subtrai vetores com vSub" $
      vSub [4, 5, 6] [1, 2, 3] `shouldBe` [3, 3, 3]

    it "escala vetor com vScale" $
      vScale 2 [1, -3, 4] `shouldBe` [2, -6, 8]

    it "calcula produto elemento a elemento com hadamard" $
      hadamard [2, 3, 4] [5, -1, 0] `shouldBe` [10, -3, 0]

  describe "operacoes com matrizes" $ do
    it "multiplica matriz por vetor com matVec" $
      matVec [[1, 2], [3, 4]] [1, 1] `shouldBe` [3, 7]

    it "transpoe matriz 2x3 com transposeM" $
      transposeM [[1, 2, 3], [4, 5, 6]] `shouldBe` [[1, 4], [2, 5], [3, 6]]

    it "calcula produto externo com outer mantendo length a linhas" $
      outer [1, 2] [3, 4, 5] `shouldBe` [[3, 4, 5], [6, 8, 10]]

    it "soma matrizes com mAdd" $
      mAdd [[1, 2], [3, 4]] [[5, 6], [7, 8]] `shouldBe` [[6, 8], [10, 12]]

    it "escala matriz com mScale" $
      mScale 3 [[1, -2], [0, 4]] `shouldBe` [[3, -6], [0, 12]]

  describe "argmax" $ do
    it "devolve o menor indice em caso de empate" $
      argmax [1, 3, 3] `shouldBe` 1

    it "encontra o maior elemento mesmo com valores negativos" $
      argmax [-10, -2, -5] `shouldBe` 1

    it "funciona para vetor unitario" $
      argmax [42] `shouldBe` 0

  describe "operacoes sobre Network" $ do
    it "zeroNetwork mantem o formato e zera todos os pesos e biases" $ do
      let z = zeroNetwork sampleNet
      z
        `shouldBe` Network
          [ Layer {weights = [[0, 0], [0, 0]], biases = [0, 0]},
            Layer {weights = [[0, 0]], biases = [0]}
          ]

    it "addNetworks com zeroNetwork devolve a propria rede" $
      addNetworks sampleNet (zeroNetwork sampleNet) `shouldBe` sampleNet

    it "scaleNetwork multiplica todos os pesos e biases por k" $
      scaleNetwork 2 sampleNet
        `shouldBe` Network
          [ Layer {weights = [[2.0, 4.0], [6.0, 8.0]], biases = [1.0, -1.0]},
            Layer {weights = [[4.0, -2.0]], biases = [2.0]}
          ]

    it "zipNetworkWith combina duas redes elemento a elemento" $
      zipNetworkWith (-) sampleNet sampleNet `shouldBe` zeroNetwork sampleNet