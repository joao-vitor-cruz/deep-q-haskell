module DQN.TypesSpec (spec) where

import DQN.Types
import Test.Hspec

spec :: Spec
spec = describe "DQN.Types" $ do
  it "actionToIndex e indexToAction sao inversas" $
    map (indexToAction . actionToIndex) [minBound .. maxBound] `shouldBe` [PushLeft, PushRight]

  it "PushLeft eh o indice 0 e PushRight eh o indice 1" $ do
    actionToIndex PushLeft `shouldBe` 0
    actionToIndex PushRight `shouldBe` 1

  it "ha 2 acoes e o estado tem 4 componentes" $ do
    numActions `shouldBe` 2
    stateSize `shouldBe` 4

  it "a rede padrao comeca com stateSize entradas e termina com numActions saidas" $ do
    let sizes = hpLayerSizes defaultHyperparams
    head sizes `shouldBe` stateSize
    last sizes `shouldBe` numActions

  it "os hiperparametros padrao sao coerentes" $ do
    let hp = defaultHyperparams
    hpGamma hp `shouldSatisfy` (\g -> g > 0 && g < 1)
    hpEpsilonEnd hp `shouldSatisfy` (< hpEpsilonStart hp)
    hpEpsilonEnd hp `shouldSatisfy` (> 0)
    hpWarmupSteps hp `shouldSatisfy` (>= hpBatchSize hp)
    hpBufferCapacity hp `shouldSatisfy` (>= hpWarmupSteps hp)
