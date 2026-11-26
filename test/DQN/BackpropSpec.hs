module DQN.BackpropSpec (spec) where

import DQN.Backprop
import DQN.Network (forward, initNetwork)
import DQN.Types
import System.Random (mkStdGen)
import Test.Hspec

-- | Testes da Parte 4 (perda e gradientes).
spec :: Spec
spec = describe "DQN.Backprop" $ do
  describe "huberLoss" $ do
    it "e quadratica dentro de [-1, 1]" $ do
      huberLoss 0.5 0 `shouldBe` 0.125
      huberLoss 2 2.5 `shouldBe` 0.125
    it "e linear fora de [-1, 1]" $ do
      huberLoss 3 0 `shouldBe` 2.5
      huberLoss 0 3 `shouldBe` 2.5
    it "vale 0 quando previsao e alvo sao iguais" $
      huberLoss 7 7 `shouldBe` 0

  describe "huberLossDerivative" $ do
    it "e igual a d dentro de [-1, 1]" $ do
      huberLossDerivative 0.5 0 `shouldBe` 0.5
      huberLossDerivative 0 0.25 `shouldBe` (-0.25)
    it "fica limitada em -1 e 1 fora do intervalo" $ do
      huberLossDerivative 50 0 `shouldBe` 1
      huberLossDerivative 0 50 `shouldBe` (-1)

  describe "backprop / sampleGradient" $ do
    it "bate com o exemplo feito a mao na teoria (rede [1,2,2])" $ do
      let (grad, loss) = sampleGradient handNet [1] 0 3.5
      loss `shouldBe` 1.0
      grad
        `shouldBe` Network
          [ Layer {weights = [[-1], [0]], biases = [-1, 0]},
            Layer {weights = [[-2, 0], [0, 0]], biases = [-1, 0]}
          ]

    it "devolve um gradiente com o mesmo formato da rede" $ do
      let (grad, _) = sampleGradient smallNet input 1 0
      shapeOf grad `shouldBe` shapeOf smallNet

    it "da gradiente zero quando a previsao ja e igual ao alvo" $ do
      let qa = forward smallNet input !! 0
          (grad, loss) = sampleGradient smallNet input 0 qa
      loss `shouldBe` 0
      allParams grad `shouldSatisfy` all (== 0)

    it "bate com o gradiente numerico (erro dentro de [-1, 1])" $ do
      let qa = forward smallNet input !! 1
      numericCheck smallNet input 1 (qa + 0.5)

    it "bate com o gradiente numerico (erro fora de [-1, 1])" $ do
      let qa = forward smallNet input !! 0
      numericCheck smallNet input 0 (qa - 3)

  describe "batchGradient" $ do
    it "com uma amostra e igual ao sampleGradient" $
      batchGradient smallNet [(input, 1, 2.0)] `shouldBe` sampleGradient smallNet input 1 2.0

    it "faz a media dos gradientes e das perdas" $ do
      let (g1, l1) = sampleGradient smallNet input 0 1.0
          (g2, l2) = sampleGradient smallNet input2 1 (-1.0)
          (gBatch, lBatch) = batchGradient smallNet [(input, 0, 1.0), (input2, 1, -1.0)]
          expected = zipWith (\a b -> (a + b) / 2) (allParams g1) (allParams g2)
      lBatch `shouldSatisfy` closeTo ((l1 + l2) / 2)
      and (zipWith closeTo expected (allParams gBatch)) `shouldBe` True

-- ---------------------------------------------------------------------------
-- Redes e entradas usadas nos testes
-- ---------------------------------------------------------------------------

-- | Rede do exemplo da teoria: 1 entrada, 2 ocultos (ReLU), 2 saídas, biases zero.
handNet :: Network
handNet =
  Network
    [ Layer {weights = [[2], [-1]], biases = [0, 0]},
      Layer {weights = [[1, 3], [0.5, 1]], biases = [0, 0]}
    ]

-- | Rede pequena com duas camadas ocultas, para a verificação numérica.
smallNet :: Network
smallNet = fst (initNetwork [4, 5, 3, 2] (mkStdGen 7))

input, input2 :: Vector
input = [0.3, -0.2, 0.05, 0.4]
input2 = [-0.1, 0.25, -0.3, 0.15]

-- ---------------------------------------------------------------------------
-- Verificação numérica do gradiente
-- ---------------------------------------------------------------------------

-- | Para CADA peso e bias w da rede, compara o gradiente do backprop com a
-- aproximação numérica (L(w + h) − L(w)) / h, com h = 1e-6.
numericCheck :: Network -> Vector -> Int -> Double -> Expectation
numericCheck net x a y = do
  let h = 1e-6
      lossOf n = snd (sampleGradient n x a y)
      baseLoss = lossOf net
      analytic = allParams (fst (sampleGradient net x a y))
      numeric =
        [ (lossOf (nudgeParam k h net) - baseLoss) / h
        | k <- [0 .. length analytic - 1]
        ]
      bad = [(k, an, nu) | (k, an, nu) <- zip3 [0 :: Int ..] analytic numeric, abs (an - nu) > 1e-4]
  bad `shouldBe` []

-- | Todos os pesos e biases da rede numa lista só, camada por camada
-- (primeiro os pesos linha a linha, depois os biases).
allParams :: Network -> [Double]
allParams (Network ls) = concat [concat w ++ b | Layer w b <- ls]

-- | Soma h ao k-ésimo parâmetro, na mesma ordem de 'allParams'.
nudgeParam :: Int -> Double -> Network -> Network
nudgeParam k h (Network ls) = Network (go k ls)
  where
    go _ [] = []
    go i (Layer w b : rest)
      | i < nW = Layer (nudgeMatrix i w) b : rest
      | i < nW + length b = Layer w (nudgeList (i - nW) b) : rest
      | otherwise = Layer w b : go (i - nW - length b) rest
      where
        nW = sum (map length w)
    nudgeList i xs = [if j == i then v + h else v | (j, v) <- zip [0 ..] xs]
    nudgeMatrix i rows = chunk (numCols rows) (nudgeList i (concat rows))
    chunk _ [] = []
    chunk n xs = take n xs : chunk n (drop n xs)

-- | Formato da rede: dimensões de cada matriz de pesos e de cada vetor de biases.
shapeOf :: Network -> [(Int, Int, Int)]
shapeOf (Network ls) = [(length w, numCols w, length b) | Layer w b <- ls]

-- | Número de colunas de uma matriz (0 se não tiver linhas).
numCols :: Matrix -> Int
numCols [] = 0
numCols (row : _) = length row

closeTo :: Double -> Double -> Bool
closeTo a b = abs (a - b) < 1e-12
