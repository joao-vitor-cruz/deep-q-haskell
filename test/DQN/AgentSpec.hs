module DQN.AgentSpec (spec) where

import DQN.Agent (tdTarget, train, trainStep)
import DQN.Optimizer (initAdam)
import DQN.Replay (emptyBuffer)
import DQN.Types
  ( Action (..),
    Agent (..),
    EpisodeLog (..),
    Hyperparams (..),
    Layer (..),
    Network (..),
    Transition (..),
    defaultHyperparams,
    numActions,
    stateSize,
  )
import System.Random (mkStdGen)
import Test.Hspec

zeroNetwork :: Network
zeroNetwork =
  Network
    [ Layer
        { weights = replicate numActions (replicate stateSize 0),
          biases = replicate numActions 0
        }
    ]

targetNetwork :: Network
targetNetwork =
  Network
    [ Layer
        { weights = [[1, 0, 0, 0], [0, 2, 0, 0]],
          biases = [0, 0]
        }
    ]

mkAgent :: Network -> Agent
mkAgent network =
  Agent
    { agentOnline = network,
      agentTarget = network,
      agentBuffer = emptyBuffer 32,
      agentOptimizer = initAdam network,
      agentTotalSteps = 0,
      agentBest = network,
      agentBestScore = 0
    }

sampleTransition :: Transition
sampleTransition =
  Transition
    { tState = [1, 0, 0, 0],
      tAction = PushLeft,
      tReward = 0.5,
      tNextState = [0, 0, 0, 0],
      tDone = True
    }

spec :: Spec
spec =
  describe "DQN.Agent" $ do
    it "tdTarget devolve apenas a recompensa em uma transicao terminal" $ do
      let terminal = sampleTransition {tReward = 1.25, tDone = True}
      tdTarget 0.9 targetNetwork terminal `shouldBe` 1.25

    it "tdTarget usa o maior Q da target network em uma transicao nao terminal" $ do
      let continuing =
            sampleTransition
              { tReward = 1,
                tNextState = [1, 3, 0, 0],
                tDone = False
              }
      tdTarget 0.5 targetNetwork continuing `shouldBe` 4

    it "treinar repetidamente no mesmo minibatch reduz a perda" $ do
      let hp = defaultHyperparams {hpLearningRate = 0.01}
          agent = mkAgent zeroNetwork
          (_, initialLoss) = trainStep hp agent [sampleTransition]
          step currentAgent = fst (trainStep hp currentAgent [sampleTransition])
          trainedAgent = iterate step agent !! 40
          (_, finalLoss) = trainStep hp trainedAgent [sampleTransition]
      finalLoss `shouldSatisfy` (< initialLoss)

    it "trainStep preserva a target network, o buffer e o contador de passos" $ do
      let agent = mkAgent zeroNetwork
          (updated, _) = trainStep defaultHyperparams agent [sampleTransition]
      agentTarget updated `shouldBe` agentTarget agent
      agentBuffer updated `shouldBe` agentBuffer agent
      agentTotalSteps updated `shouldBe` agentTotalSteps agent

    it "train devolve os logs dos episodios em ordem e com numeracao crescente" $ do
      let hp =
            defaultHyperparams
              { hpLayerSizes = [stateSize, 8, numActions],
                hpBatchSize = 4,
                hpBufferCapacity = 64,
                hpWarmupSteps = 10,
                hpTargetUpdateEvery = 5,
                hpMaxStepsPerEpisode = 8,
                hpEpisodes = 5
              }
          (_, logs) = train hp (mkStdGen 42)
      map logEpisode logs `shouldBe` [1, 2, 3, 4, 5]
