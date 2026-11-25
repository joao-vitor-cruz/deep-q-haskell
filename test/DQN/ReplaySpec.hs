module DQN.ReplaySpec (spec) where

import Data.Foldable (toList)
import DQN.Replay (bufferSize, emptyBuffer, pushTransition, sampleBatch)
import DQN.Types (Action (..), ReplayBuffer (..), Transition (..))
import System.Random (mkStdGen)
import Test.Hspec

-- | Transição de teste identificada pelo número k (fica na recompensa),
-- para conseguirmos saber qual transição é qual.
mkTransition :: Int -> Transition
mkTransition k =
  Transition
    { tState = [fromIntegral k, 0, 0, 0],
      tAction = if even k then PushLeft else PushRight,
      tReward = fromIntegral k,
      tNextState = [fromIntegral k + 1, 0, 0, 0],
      tDone = False
    }

-- | Insere as transições 1..n, nessa ordem, num buffer com a capacidade dada.
fillBuffer :: Int -> Int -> ReplayBuffer
fillBuffer capacity n = foldl (flip pushTransition) (emptyBuffer capacity) (map mkTransition [1 .. n])

-- | Testes da Parte 5 (experience replay).
spec :: Spec
spec =
  describe "DQN.Replay" $ do
    it "emptyBuffer comeca vazio e guarda a capacidade" $ do
      let b = emptyBuffer 10
      bufferSize b `shouldBe` 0
      bufferCapacity b `shouldBe` 10

    it "pushTransition aumenta o tamanho em 1 enquanto ha espaco" $ do
      bufferSize (fillBuffer 10 1) `shouldBe` 1
      bufferSize (fillBuffer 10 4) `shouldBe` 4

    it "o tamanho nunca passa da capacidade" $ do
      let tamanhos = [bufferSize (fillBuffer 5 n) | n <- [0 .. 20]]
      all (<= 5) tamanhos `shouldBe` True
      bufferSize (fillBuffer 5 20) `shouldBe` 5

    it "com capacidade 3 e 5 insercoes, sobram as 3 ultimas (FIFO)" $ do
      let b = fillBuffer 3 5
      toList (bufferItems b) `shouldBe` map mkTransition [3, 4, 5]

    it "sampleBatch 32 devolve 32 transicoes, todas presentes no buffer" $ do
      let b = fillBuffer 100 50
          (amostra, _) = sampleBatch 32 b (mkStdGen 42)
          presentes = toList (bufferItems b)
      length amostra `shouldBe` 32
      all (`elem` presentes) amostra `shouldBe` True

    it "sampleBatch funciona com reposicao (pede mais itens do que o buffer tem)" $ do
      let b = fillBuffer 10 3
          (amostra, _) = sampleBatch 32 b (mkStdGen 7)
      length amostra `shouldBe` 32
      all (`elem` toList (bufferItems b)) amostra `shouldBe` True

    it "sampleBatch nao sorteia sempre a mesma transicao" $ do
      let b = fillBuffer 100 100
          (amostra, _) = sampleBatch 32 b (mkStdGen 1)
      -- Pelo menos um par de sorteios seguidos deu transicoes diferentes.
      or (zipWith (/=) amostra (drop 1 amostra)) `shouldBe` True

    it "o StdGen devolvido e diferente do recebido" $ do
      let g = mkStdGen 42
          (_, g') = sampleBatch 32 (fillBuffer 10 10) g
      show g' `shouldNotBe` show g

    it "sampleBatch 0 devolve lista vazia" $ do
      let (amostra, _) = sampleBatch 0 (fillBuffer 10 5) (mkStdGen 3)
      amostra `shouldBe` []
