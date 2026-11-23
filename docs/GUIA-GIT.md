# Guia de Git e GitHub do grupo

Regra do professor: **no mínimo 2 commits por integrante, identificáveis pelo
autor**. Se o seu commit não aparecer com o seu nome no GitHub, ele não conta.

## 1. Configuração (uma vez por computador)

```bash
git config --global user.name "Seu Nome Completo"
git config --global user.email "o-email-da-sua-conta-do-github@exemplo.com"
```

O e-mail **precisa ser o mesmo da sua conta do GitHub**. Confira em
GitHub → Settings → Emails.

Aceite o convite de colaborador do repositório (chega por e-mail ou aparece
em github.com/notifications).

## 2. Baixar o projeto

```bash
git clone <URL-DO-REPOSITORIO>
cd deep-q-haskell
cabal build && cabal test
```

## 3. Fazer a sua parte

```bash
git checkout main
git pull                          # pega o que já entrou
git checkout -b parte-1-env       # troque pelo número e nome da sua parte
```

Edite **só** os arquivos da sua parte (AGENTS.md, seção 2.2). Antes de
commitar:

```bash
cabal build && cabal test         # precisa passar
git status                        # confira se só mudou o que é seu
```

### Mensagem de commit: título + descrição

Todo commit tem **título e corpo** descrevendo o que foi feito, como e como
foi testado (regras e exemplo completo no AGENTS.md, seção 2.6). Peça para
a sua IA escrever a mensagem a partir do que vocês fizeram e revise antes
de usar.

Jeito mais fácil: salve a mensagem num arquivo **fora** da pasta do
projeto (por exemplo `../msg.txt`) e use `git commit -F`.

### Commit de código

```bash
git add src/DQN/Env.hs test/DQN/EnvSpec.hs
git commit -F ../msg.txt
```

### Commit de teoria

```bash
git add docs/teoria/01-fundamentos-rl.md
git commit -F ../msg.txt
```

Sem arquivo, `git commit` (sem `-m`) abre um editor no terminal para
escrever título e corpo.

### Enviar e abrir o Pull Request

```bash
git push -u origin parte-1-env
```

No GitHub, clique em **Compare & pull request**, preencha o checklist e
aguarde a CI ficar verde e a aprovação.

## 4. Atualizar a sua branch com a main

Quando outra parte entrar e você precisar dela:

```bash
git checkout main && git pull
git checkout parte-1-env
git merge main
```

Como cada um mexe em arquivos diferentes, conflitos devem ser raros. Se
aparecer um, **não apague código dos outros**: peça ajuda no grupo.

## 5. Não faça

- Commitar direto na `main` (ela é protegida).
- Pedir para outra pessoa commitar por você: o commit fica no nome dela.
- Alterar `Types.hs`, o `.cabal` ou a parte de outra pessoa.
- Deixar tudo para o último dia.

## 6. Usando IA

O professor pediu que cada um use a própria IA. Para todas seguirem o mesmo
padrão, as regras estão no `AGENTS.md`.

- **Claude Code, Cursor, GitHub Copilot, Codex, Gemini CLI:** abra a pasta do
  projeto. Eles leem o `AGENTS.md` (ou o `CLAUDE.md`/`GEMINI.md`, que apontam
  para ele) sozinhos. Comece dizendo: "Sou responsável pela Parte N".
- **ChatGPT, Claude ou Gemini no navegador:** cole o conteúdo inteiro do
  `AGENTS.md` na primeira mensagem e diga qual é a sua parte. Cole também o
  `src/DQN/Types.hs`.

Lembre: **8 dos 10 pontos são arguição.** Peça para a IA explicar o código
até você conseguir explicar sem ela.
