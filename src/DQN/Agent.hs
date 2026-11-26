{-# LANGUAGE BangPatterns #-}

-- |
-- Module      : DQN.Agent
-- Description : Parte 7 — o algoritmo DQN completo.
module DQN.Agent (newAgent, tdTarget, trainStep, runEpisode, train) where

import Control.DeepSeq (force)
import DQN.Backprop (batchGradient)
import DQN.Env (resetEnv, stateToVector, stepEnv)
import DQN.Network (forward, initNetwork)
import DQN.Optimizer (adamStep, initAdam)
import DQN.Policy (epsilonAt, epsilonGreedy)
import DQN.Replay (bufferSize, emptyBuffer, pushTransition, sampleBatch)
import DQN.Types
  ( Agent (..),
    EpisodeLog (..),
    Hyperparams (..),
    Network,
    StepResult (..),
    Transition (..),
    actionToIndex,
  )
import System.Random (StdGen)

-- | Cria o agente inicial: inicializa a rede online, copia-a para a target,
-- zera o otimizador e o contador de passos e cria o replay buffer vazio.
newAgent :: Hyperparams -> StdGen -> (Agent, StdGen)
newAgent hp gen =
  let (online, gen') = initNetwork (hpLayerSizes hp) gen
      !agent =
        force
          ( Agent
            { agentOnline = online,
              agentTarget = online,
              agentBuffer = emptyBuffer (hpBufferCapacity hp),
              agentOptimizer = initAdam online,
              agentTotalSteps = 0,
              agentBest = online,
              agentBestScore = 0
            }
          )
   in (agent, gen')

-- | Calcula o alvo TD usando exclusivamente a rede target:
-- @y = r@ se a transição terminou; caso contrário,
-- @y = r + gamma * max_a' Q_target(s', a')@.
tdTarget :: Double -> Network -> Transition -> Double
tdTarget gamma target transition
  | tDone transition = tReward transition
  | otherwise =
      case forward target (tNextState transition) of
        [] -> error "DQN.Agent.tdTarget: target network has no actions"
        nextValues -> tReward transition + gamma * maximum nextValues

-- | Atualiza a rede online e o estado do Adam com um minibatch. A rede target,
-- o replay buffer e o contador de passos permanecem inalterados.
--
-- Pré-condição: o minibatch não pode estar vazio.
trainStep :: Hyperparams -> Agent -> [Transition] -> (Agent, Double)
trainStep hp agent transitions
  | null transitions = error "DQN.Agent.trainStep: empty minibatch"
  | otherwise =
      let online = agentOnline agent
          examples =
            [ ( tState transition,
                actionToIndex (tAction transition),
                tdTarget (hpGamma hp) (agentTarget agent) transition
              )
              | transition <- transitions
            ]
          (gradient, averageLoss) = batchGradient online examples
          (online', optimizer') =
            adamStep (hpLearningRate hp) (agentOptimizer agent) online gradient
          !agent' =
            force
              ( agent
                  { agentOnline = online',
                    agentOptimizer = optimizer'
                  }
              )
       in (agent', averageLoss)

-- | Executa um episódio até o ambiente terminar ou atingir o limite de passos.
-- Em cada passo coleta uma transição, treina após o warmup e sincroniza a rede
-- target nos intervalos configurados.
runEpisode :: Hyperparams -> Int -> Agent -> StdGen -> (Agent, EpisodeLog, StdGen)
runEpisode hp episodeNumber agent0 gen0 =
  let (initialState, gen1) = resetEnv gen0
   in go False initialState agent0 gen1 0 0 0 0
  where
    go !finished !state !agent !gen !episodeSteps !rewardSum !lossSum !trainCount
      | finished || episodeSteps >= hpMaxStepsPerEpisode hp =
          let averageLoss =
                if trainCount == 0
                  then 0
                  else lossSum / fromIntegral trainCount
              !finalAgent = force agent
              episodeLog =
                EpisodeLog
                  { logEpisode = episodeNumber,
                    logTotalReward = rewardSum,
                    logEpsilon = epsilonAt hp (agentTotalSteps finalAgent),
                    logAvgLoss = averageLoss
                  }
           in (finalAgent, episodeLog, gen)
      | otherwise =
          let epsilon = epsilonAt hp (agentTotalSteps agent)
              (action, genAfterAction) =
                epsilonGreedy epsilon (agentOnline agent) (stateToVector state) gen
              result = stepEnv state action
              transition =
                Transition
                  { tState = stateToVector state,
                    tAction = action,
                    tReward = reward result,
                    tNextState = stateToVector (nextState result),
                    tDone = done result
                  }
              nextStep = agentTotalSteps agent + 1
              !agentWithTransition =
                force
                  ( agent
                      { agentBuffer = pushTransition transition (agentBuffer agent),
                        agentTotalSteps = nextStep
                      }
                  )
              currentBuffer = agentBuffer agentWithTransition
              currentBufferSize = bufferSize currentBuffer
              shouldTrain =
                hpBatchSize hp > 0
                  && currentBufferSize > 0
                  && currentBufferSize >= hpWarmupSteps hp
              (batch, genAfterSampling) =
                if shouldTrain
                  then sampleBatch (hpBatchSize hp) currentBuffer genAfterAction
                  else ([], genAfterAction)
              (agentAfterTrain, loss, didTrain) =
                case batch of
                  [] -> (agentWithTransition, 0, False)
                  _ ->
                    let (updatedAgent, stepLoss) = trainStep hp agentWithTransition batch
                     in (updatedAgent, stepLoss, True)
              updateEvery = hpTargetUpdateEvery hp
              agentAfterSync
                | updateEvery > 0 && nextStep `mod` updateEvery == 0 =
                    agentAfterTrain {agentTarget = agentOnline agentAfterTrain}
                | otherwise = agentAfterTrain
              !strictAgentAfterSync = force agentAfterSync
              nextLossSum = if didTrain then lossSum + loss else lossSum
              nextTrainCount = if didTrain then trainCount + 1 else trainCount
           in go
                (done result)
                (nextState result)
                strictAgentAfterSync
                genAfterSampling
                (episodeSteps + 1)
                (rewardSum + reward result)
                nextLossSum
                nextTrainCount

-- | Treina um agente por @hpEpisodes@ episódios e devolve o agente final e os
-- logs em ordem. Guarda a rede online sempre que a média da última janela
-- configurada supera a melhor média registrada.
train :: Hyperparams -> StdGen -> (Agent, [EpisodeLog])
train hp gen =
  let (agent0, gen1) = newAgent hp gen
   in go 1 agent0 gen1 [] []
  where
    window = hpBestWindow hp

    go !episode !agent !currentGen !recentRewards !logs
      | episode > hpEpisodes hp = (agent, reverse logs)
      | otherwise =
          let (episodeAgent, episodeLog, nextGen) =
                runEpisode hp episode agent currentGen
              rewards =
                take (max 0 window) (logTotalReward episodeLog : recentRewards)
              hasFullWindow = window > 0 && length rewards >= window
              meanReward =
                if hasFullWindow
                  then sum rewards / fromIntegral window
                  else 0
              bestAgent
                | hasFullWindow && meanReward > agentBestScore episodeAgent =
                    episodeAgent
                      { agentBest = agentOnline episodeAgent,
                        agentBestScore = meanReward
                      }
                | otherwise = episodeAgent
              !agent' = force bestAgent
           in go (episode + 1) agent' nextGen rewards (episodeLog : logs)
