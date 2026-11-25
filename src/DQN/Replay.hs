-- |
-- Module      : DQN.Replay
-- Description : Parte 5 — experience replay.
--
-- O replay buffer guarda as últimas experiências (transições) vividas pelo
-- agente. Em vez de treinar a rede com as transições na ordem em que
-- acontecem (muito parecidas entre si, pois são passos seguidos do mesmo
-- episódio), o agente sorteia um minibatch aleatório deste buffer. Isso
-- quebra a correlação entre as amostras e permite reaproveitar cada
-- experiência várias vezes.
--
-- O buffer é uma fila FIFO com capacidade máxima: quando está cheio, a
-- transição mais antiga sai para a nova entrar. Usamos 'Seq' (de
-- "Data.Sequence") porque ela permite inserir no fim, remover do começo e
-- acessar por índice de forma eficiente.
module DQN.Replay (emptyBuffer, pushTransition, bufferSize, sampleBatch) where

import Data.Sequence ((|>))
import qualified Data.Sequence as Seq
import DQN.Types (ReplayBuffer (..), Transition)
import System.Random (StdGen, randomR)

-- | Buffer vazio com a capacidade dada (número máximo de transições).
emptyBuffer :: Int -> ReplayBuffer
emptyBuffer capacity =
  ReplayBuffer
    { bufferCapacity = capacity,
      bufferItems = Seq.empty
    }

-- | Adiciona uma transição no FIM do buffer. Se o buffer já está cheio,
-- primeiro remove a transição MAIS ANTIGA (a do começo), como numa fila FIFO.
-- Assim o tamanho nunca passa da capacidade.
--
-- Se a capacidade for menor ou igual a 0, nada é guardado.
pushTransition :: Transition -> ReplayBuffer -> ReplayBuffer
pushTransition transition buffer
  | capacity <= 0 = buffer
  | otherwise = buffer {bufferItems = semMaisAntiga |> transition}
  where
    capacity = bufferCapacity buffer
    items = bufferItems buffer
    -- Cheio: descarta o primeiro elemento (o mais antigo). Senão, mantém tudo.
    semMaisAntiga
      | Seq.length items >= capacity = Seq.drop 1 items
      | otherwise = items

-- | Quantas transições há no buffer agora.
bufferSize :: ReplayBuffer -> Int
bufferSize = Seq.length . bufferItems

-- | Sorteia n transições de forma uniforme, COM reposição (a mesma transição
-- pode sair mais de uma vez). Cada sorteio escolhe um índice i uniforme em
-- [0, tamanho − 1] e usa o 'StdGen' devolvido pelo sorteio anterior, para que
-- os índices sejam diferentes entre si. Devolve as transições e o 'StdGen' novo.
--
-- Pré-condição: buffer não vazio (com buffer vazio não há o que sortear).
-- Se n <= 0, devolve a lista vazia e o mesmo 'StdGen'.
sampleBatch :: Int -> ReplayBuffer -> StdGen -> ([Transition], StdGen)
sampleBatch n buffer gen
  | n <= 0 = ([], gen)
  | otherwise = (transition : resto, genFinal)
  where
    items = bufferItems buffer
    -- Índice válido vai de 0 até tamanho − 1 (o último índice NÃO é o tamanho).
    (i, gen') = randomR (0, Seq.length items - 1) gen
    transition = Seq.index items i
    -- Os outros n − 1 sorteios continuam a partir do gerador novo.
    (resto, genFinal) = sampleBatch (n - 1) buffer gen'
