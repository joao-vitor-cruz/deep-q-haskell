{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

-- |
-- Module      : DQN.Types
-- Description : Tipos compartilhados por todos os módulos do projeto.
--
-- ESTE ARQUIVO É O CONTRATO DO PROJETO.
--
-- Todos os módulos usam os tipos definidos aqui. Nenhum módulo deve criar
-- um tipo próprio para representar algo que já existe neste arquivo.
--
-- Mudanças neste arquivo só via Pull Request aprovado pelo grupo, porque
-- alterar um tipo aqui quebra o código de todo mundo.
module DQN.Types
  ( -- * Álgebra linear
    Vector,
    Matrix,

    -- * Ambiente (CartPole)
    CartState (..),
    Action (..),
    StepResult (..),
    actionToIndex,
    indexToAction,
    numActions,
    stateSize,

    -- * Rede neural
    Layer (..),
    Network (..),
    LayerTrace (..),

    -- * Otimizador
    AdamState (..),

    -- * Experiência e replay buffer
    Transition (..),
    ReplayBuffer (..),

    -- * Agente e treinamento
    Hyperparams (..),
    defaultHyperparams,
    Agent (..),
    EpisodeLog (..),
  )
where

import Control.DeepSeq (NFData)
import Data.Sequence (Seq)
import GHC.Generics (Generic)

-- ---------------------------------------------------------------------------
-- Álgebra linear
-- ---------------------------------------------------------------------------

-- | Vetor de números reais.
type Vector = [Double]

-- | Matriz armazenada por linhas: uma lista de linhas, cada linha um 'Vector'.
--
-- Uma matriz de dimensão m x n é uma lista com m linhas de n elementos.
type Matrix = [[Double]]

-- ---------------------------------------------------------------------------
-- Ambiente: CartPole
-- ---------------------------------------------------------------------------

-- | Estado do CartPole: um carrinho que anda num trilho com um pêndulo
-- (haste) preso em cima. O objetivo é manter a haste em pé.
--
-- O estado é CONTÍNUO (quatro números reais). É por isso que uma tabela Q
-- não serve e precisamos de uma rede neural para aproximar Q(s, a).
data CartState = CartState
  { -- | Posição do carrinho no trilho (metros). 0 é o centro.
    cartPosition :: !Double,
    -- | Velocidade do carrinho (m/s).
    cartVelocity :: !Double,
    -- | Ângulo da haste em relação à vertical (radianos). 0 é em pé.
    poleAngle :: !Double,
    -- | Velocidade angular da haste (rad/s).
    poleAngularVelocity :: !Double
  }
  deriving (Show, Eq, Generic, NFData)

-- | Ações possíveis: empurrar o carrinho para a esquerda ou para a direita.
data Action = PushLeft | PushRight
  deriving (Show, Eq, Ord, Enum, Bounded, Generic, NFData)

-- | Resultado de aplicar uma ação no ambiente.
data StepResult = StepResult
  { -- | Estado seguinte.
    nextState :: !CartState,
    -- | Recompensa recebida nesta transição.
    reward :: !Double,
    -- | 'True' se o episódio terminou (haste caiu ou carrinho saiu do trilho).
    done :: !Bool
  }
  deriving (Show, Eq, Generic, NFData)

-- | Converte uma ação no índice da saída da rede que a representa.
--
-- A saída da rede é um vetor com um Q-valor por ação, nesta ordem:
-- índice 0 = 'PushLeft', índice 1 = 'PushRight'.
actionToIndex :: Action -> Int
actionToIndex = fromEnum

-- | Inverso de 'actionToIndex'.
indexToAction :: Int -> Action
indexToAction = toEnum

-- | Número de ações (tamanho da camada de saída da rede).
numActions :: Int
numActions = length [minBound .. maxBound :: Action]

-- | Tamanho do vetor de estado (tamanho da camada de entrada da rede).
stateSize :: Int
stateSize = 4

-- ---------------------------------------------------------------------------
-- Rede neural
-- ---------------------------------------------------------------------------

-- | Uma camada totalmente conectada (densa).
--
-- Se a camada recebe n entradas e produz m saídas:
--
-- * 'weights' é uma matriz m x n (m linhas, uma por neurônio de saída);
-- * 'biases' é um vetor com m elementos.
--
-- A saída antes da ativação é: z = weights * x + biases.
data Layer = Layer
  { weights :: Matrix,
    biases :: Vector
  }
  deriving (Show, Eq, Generic, NFData)

-- | Rede neural: sequência de camadas, da entrada para a saída.
--
-- Convenção de ativação (vale para o projeto todo):
--
-- * todas as camadas ocultas usam ReLU;
-- * a ÚLTIMA camada é linear (sem ativação), porque Q-valores podem ser
--   qualquer número real.
--
-- O mesmo tipo é usado para guardar GRADIENTES: um gradiente tem exatamente
-- o mesmo formato da rede (uma matriz e um vetor por camada).
newtype Network = Network {layers :: [Layer]}
  deriving (Show, Eq, Generic)

instance NFData Network

-- | Valores intermediários de uma camada durante o forward pass,
-- guardados porque o backpropagation precisa deles.
data LayerTrace = LayerTrace
  { -- | Entrada que a camada recebeu.
    traceInput :: Vector,
    -- | Saída antes da ativação (z = W x + b).
    tracePreActivation :: Vector,
    -- | Saída depois da ativação (igual a z na última camada).
    traceOutput :: Vector
  }
  deriving (Show, Eq, Generic, NFData)

-- ---------------------------------------------------------------------------
-- Otimizador
-- ---------------------------------------------------------------------------

-- | Estado do otimizador Adam.
--
-- O Adam guarda, para CADA peso da rede, uma média móvel do gradiente ('adamM')
-- e uma média móvel do gradiente ao quadrado ('adamV'). Por isso os dois
-- campos têm o mesmo formato da rede ('Network').
--
-- Nos testes, SGD puro não fez o DQN aprender no CartPole em tempo razoável;
-- com Adam ele aprende. Por isso o projeto usa Adam.
data AdamState = AdamState
  { -- | Primeiro momento (média móvel dos gradientes).
    adamM :: Network,
    -- | Segundo momento (média móvel dos gradientes ao quadrado).
    adamV :: Network,
    -- | Quantas atualizações já foram feitas (começa em 0).
    adamStepCount :: !Int
  }
  deriving (Show, Eq, Generic, NFData)

-- ---------------------------------------------------------------------------
-- Experiência e replay buffer
-- ---------------------------------------------------------------------------

-- | Uma experiência (s, a, r, s', done) vivida pelo agente.
--
-- Os estados são guardados já como 'Vector', prontos para entrar na rede.
data Transition = Transition
  { tState :: Vector,
    tAction :: Action,
    tReward :: Double,
    tNextState :: Vector,
    tDone :: Bool
  }
  deriving (Show, Eq, Generic, NFData)

-- | Memória de experiências com capacidade máxima.
--
-- Quando está cheia, a experiência MAIS ANTIGA é descartada para dar lugar
-- à nova (fila FIFO). Usamos 'Seq' porque permite acesso por índice
-- eficiente, o que importa na hora de sortear o minibatch.
data ReplayBuffer = ReplayBuffer
  { bufferCapacity :: !Int,
    bufferItems :: Seq Transition
  }
  deriving (Show, Eq, Generic, NFData)

-- ---------------------------------------------------------------------------
-- Agente e treinamento
-- ---------------------------------------------------------------------------

-- | Hiperparâmetros do treinamento.
data Hyperparams = Hyperparams
  { -- | Tamanhos das camadas, da entrada à saída. Ex.: [4, 64, 2].
    hpLayerSizes :: [Int],
    -- | Fator de desconto (gamma).
    hpGamma :: !Double,
    -- | Taxa de aprendizado.
    hpLearningRate :: !Double,
    -- | Tamanho do minibatch sorteado do replay buffer.
    hpBatchSize :: !Int,
    -- | Capacidade do replay buffer.
    hpBufferCapacity :: !Int,
    -- | Passos mínimos no buffer antes de começar a treinar.
    hpWarmupSteps :: !Int,
    -- | A cada quantos passos a target network é sincronizada.
    hpTargetUpdateEvery :: !Int,
    -- | Epsilon inicial (exploração).
    hpEpsilonStart :: !Double,
    -- | Epsilon mínimo.
    hpEpsilonEnd :: !Double,
    -- | Em quantos passos o epsilon decai do inicial até o mínimo.
    hpEpsilonDecaySteps :: !Int,
    -- | Limite de passos por episódio.
    hpMaxStepsPerEpisode :: !Int,
    -- | Quantidade de episódios de treinamento.
    hpEpisodes :: !Int,
    -- | Janela (em episódios) da média móvel usada para escolher a melhor rede.
    hpBestWindow :: !Int
  }
  deriving (Show, Eq, Generic, NFData)

-- | Valores padrão, testados para o CartPole.
defaultHyperparams :: Hyperparams
defaultHyperparams =
  Hyperparams
    { hpLayerSizes = [stateSize, 64, numActions],
      hpGamma = 0.99,
      hpLearningRate = 0.001,
      hpBatchSize = 32,
      hpBufferCapacity = 10000,
      hpWarmupSteps = 500,
      hpTargetUpdateEvery = 500,
      hpEpsilonStart = 1.0,
      hpEpsilonEnd = 0.05,
      hpEpsilonDecaySteps = 10000,
      hpMaxStepsPerEpisode = 500,
      hpEpisodes = 400,
      hpBestWindow = 20
    }

-- | Estado completo do agente DQN.
data Agent = Agent
  { -- | Rede online: é a que escolhe as ações e é treinada.
    agentOnline :: Network,
    -- | Target network: cópia congelada da online, usada para calcular o alvo.
    agentTarget :: Network,
    -- | Memória de experiências.
    agentBuffer :: ReplayBuffer,
    -- | Estado do otimizador Adam da rede online.
    agentOptimizer :: AdamState,
    -- | Total de passos dados desde o início do treinamento.
    agentTotalSteps :: !Int,
    -- | Cópia da rede online no momento em que ela teve a melhor média
    -- móvel de recompensa. O DQN é instável: pode aprender e depois piorar.
    -- Guardar a melhor rede é a forma simples de não perder o que aprendeu.
    agentBest :: Network,
    -- | Melhor média móvel de recompensa já registrada (começa em 0).
    agentBestScore :: !Double
  }
  deriving (Show, Eq, Generic, NFData)

-- | Registro de um episódio de treinamento (vira uma linha do CSV).
data EpisodeLog = EpisodeLog
  { logEpisode :: !Int,
    -- | Soma das recompensas do episódio (no CartPole, = nº de passos em pé).
    logTotalReward :: !Double,
    -- | Epsilon no fim do episódio.
    logEpsilon :: !Double,
    -- | Perda média dos passos de treino do episódio (0 se não treinou).
    logAvgLoss :: !Double
  }
  deriving (Show, Eq, Generic, NFData)
