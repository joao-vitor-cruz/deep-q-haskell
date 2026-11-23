module DQN.EnvSpec (spec) where

import DQN.Env (isTerminal, resetEnv, stateToVector, stepEnv)
import DQN.Types (Action (..), CartState (..), StepResult (..))
import System.Random (mkStdGen)
import Test.Hspec

-- | Estado perfeitamente centrado e parado.
zeroState :: CartState
zeroState =
  CartState
    { cartPosition = 0.0,
      cartVelocity = 0.0,
      poleAngle = 0.0,
      poleAngularVelocity = 0.0
    }

-- | Executa passos aplicando sempre a mesma ação até terminar ou atingir um limite.
stepsUntilDone :: Int -> CartState -> Action -> Int
stepsUntilDone maxSteps state action = go 0 state
  where
    go count current
      | count >= maxSteps = count
      | done res = count + 1
      | otherwise = go (count + 1) (nextState res)
      where
        res = stepEnv current action

-- | Testes da Parte 1 (o ambiente CartPole).
spec :: Spec
spec =
  describe "DQN.Env" $ do
    it "resetEnv gera os 4 valores dentro do intervalo [-0.05, 0.05]" $ do
      let (s, _) = resetEnv (mkStdGen 42)
          vals = stateToVector s
      length vals `shouldBe` 4
      all (\v -> v >= -0.05 && v <= 0.05) vals `shouldBe` True

    it "PushRight deixa a velocidade positiva e PushLeft deixa negativa a partir do zero" $ do
      let resRight = stepEnv zeroState PushRight
          resLeft = stepEnv zeroState PushLeft
      cartVelocity (nextState resRight) `shouldSatisfy` (> 0.0)
      cartVelocity (nextState resLeft) `shouldSatisfy` (< 0.0)
      reward resRight `shouldBe` 1.0
      reward resLeft `shouldBe` 1.0

    it "isTerminal identifica corretamente estados terminais e nao terminais" $ do
      isTerminal zeroState `shouldBe` False
      isTerminal (zeroState {poleAngle = 0.25}) `shouldBe` True
      isTerminal (zeroState {cartPosition = 2.5}) `shouldBe` True

    it "empurrar sempre para o mesmo lado encerra o episodio em menos de 100 passos" $ do
      let passosDireita = stepsUntilDone 100 zeroState PushRight
          passosEsquerda = stepsUntilDone 100 zeroState PushLeft
      passosDireita `shouldSatisfy` (< 100)
      passosEsquerda `shouldSatisfy` (< 100)

    it "stateToVector preserva a ordem [posicao, velocidade, angulo, velocidade angular]" $ do
      let s = CartState 1.1 2.2 3.3 4.4
      stateToVector s `shouldBe` [1.1, 2.2, 3.3, 4.4]