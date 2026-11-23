-- |
-- Module      : DQN.Linalg
-- Description : Parte 2 — álgebra linear e operações sobre redes.
--
-- Operações básicas sobre vetores, matrizes e redes neurais usadas pelos
-- demais módulos do agente DQN (forward pass, backpropagation e otimizador).
module DQN.Linalg
  ( dot,
    vAdd,
    vSub,
    vScale,
    hadamard,
    matVec,
    transposeM,
    outer,
    mAdd,
    mScale,
    argmax,
    zipNetworkWith,
    mapNetwork,
    addNetworks,
    scaleNetwork,
    zeroNetwork,
  )
where

import DQN.Types (Layer (..), Matrix, Network (..), Vector)
import Data.List (transpose)

-- ---------------------------------------------------------------------------
-- Operações sobre vetores
-- ---------------------------------------------------------------------------

-- | Produto escalar entre dois vetores: @Σ aᵢ · bᵢ@.
dot :: Vector -> Vector -> Double
dot a b = sum (zipWith (*) a b)

-- | Soma elemento a elemento entre dois vetores: @(a + b)ᵢ = aᵢ + bᵢ@.
vAdd :: Vector -> Vector -> Vector
vAdd = zipWith (+)

-- | Subtração elemento a elemento entre dois vetores: @(a − b)ᵢ = aᵢ − bᵢ@.
vSub :: Vector -> Vector -> Vector
vSub = zipWith (-)

-- | Multiplicação de um vetor por um escalar: @(k · v)ᵢ = k · vᵢ@.
vScale :: Double -> Vector -> Vector
vScale k = map (k *)

-- | Produto de Hadamard (elemento a elemento): @(a ⊙ b)ᵢ = aᵢ · bᵢ@.
hadamard :: Vector -> Vector -> Vector
hadamard = zipWith (*)

-- ---------------------------------------------------------------------------
-- Operações sobre matrizes
-- ---------------------------------------------------------------------------

-- | Multiplicação de matriz @m × n@ por vetor de tamanho @n@, produzindo um
-- vetor de tamanho @m@. Cada componente @i@ é o produto escalar da linha @i@
-- de @M@ com @v@: @(M · v)ᵢ = Σⱼ Mᵢⱼ · vⱼ@.
matVec :: Matrix -> Vector -> Vector
matVec m v = [dot row v | row <- m]

-- | Transposta de uma matriz: @(Mᵀ)ᵢⱼ = Mⱼᵢ@.
transposeM :: Matrix -> Matrix
transposeM = transpose

-- | Produto externo entre dois vetores: @(outer a b)ᵢⱼ = aᵢ · bⱼ@.
-- Produz uma matriz com @length a@ linhas e @length b@ colunas.
outer :: Vector -> Vector -> Matrix
outer a b = [[x * y | y <- b] | x <- a]

-- | Soma elemento a elemento entre duas matrizes: @(A + B)ᵢⱼ = Aᵢⱼ + Bᵢⱼ@.
mAdd :: Matrix -> Matrix -> Matrix
mAdd = zipWith vAdd

-- | Multiplicação de uma matriz por um escalar: @(k · M)ᵢⱼ = k · Mᵢⱼ@.
mScale :: Double -> Matrix -> Matrix
mScale k = map (vScale k)

-- ---------------------------------------------------------------------------
-- Seleção de índice
-- ---------------------------------------------------------------------------

-- | Índice do maior elemento de um vetor. Em caso de empate, devolve o menor
-- índice.
--
-- Pré-condição: o vetor não pode ser vazio.
argmax :: Vector -> Int
argmax [] = error "DQN.Linalg.argmax: vetor vazio"
argmax (x : xs) = go 0 x 1 xs
  where
    go bestIdx _ _ [] = bestIdx
    go bestIdx bestVal idx (y : ys)
      | y > bestVal = go idx y (idx + 1) ys
      | otherwise = go bestIdx bestVal (idx + 1) ys

-- ---------------------------------------------------------------------------
-- Operações sobre redes neurais
-- ---------------------------------------------------------------------------

-- | Combina duas redes de mesmo formato aplicando @f@ peso a peso e bias a bias.
zipNetworkWith :: (Double -> Double -> Double) -> Network -> Network -> Network
zipNetworkWith f (Network ls1) (Network ls2) =
  Network (zipWith zipLayer ls1 ls2)
  where
    zipLayer (Layer w1 b1) (Layer w2 b2) =
      Layer
        { weights = zipWith (zipWith f) w1 w2,
          biases = zipWith f b1 b2
        }

-- | Aplica uma função @f@ a todos os pesos e biases de uma rede.
mapNetwork :: (Double -> Double) -> Network -> Network
mapNetwork f (Network ls) = Network (map mapLayer ls)
  where
    mapLayer (Layer w b) =
      Layer
        { weights = map (map f) w,
          biases = map f b
        }

-- | Soma duas redes de mesmo formato peso a peso e bias a bias.
addNetworks :: Network -> Network -> Network
addNetworks = zipNetworkWith (+)

-- | Multiplica todos os pesos e biases de uma rede por um escalar @k@.
scaleNetwork :: Double -> Network -> Network
scaleNetwork k = mapNetwork (k *)

-- | Cria uma rede com o mesmo formato da rede recebida, mas com todos os
-- pesos e biases zerados.
zeroNetwork :: Network -> Network
zeroNetwork = mapNetwork (const 0)