# Setup Git + VS Code su Mac

Serve il Terminale (`Cmd + Spazio` → "Terminale").

## 1. Installa Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Alla fine copia e incolla le due righe che Homebrew ti mostra (`eval ...`). Verifica con `brew --version`.

## 2. Installa git, GitHub CLI e VS Code

```bash
brew install git gh
brew install --cask visual-studio-code
```

## 3. Configura la tua identità

Usa la stessa email del tuo account GitHub:

```bash
git config --global user.name "Nome Cognome"
git config --global user.email "tua@email.com"
git config --global init.defaultBranch main
git config --global credential.helper osxkeychain
```

## 4. Accedi a GitHub con gh

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
```

Scegli *Login with a web browser*, copia il codice nel Terminale e conferma nel browser.

## 5. Clona la repository

```bash
cd ~/Documents
gh repo clone DinuIustinAdrian/final_project_data_science
```

Se non riesce, chiedi a DinuIustinAdrian di invitarti come collaboratore e accetta l'invito dalla email o da github.com/notifications.

## 6. Collega VS Code

1. Apri VS Code.
2. `Cmd + Shift + P` → scrivi `Shell Command: Install 'code' command in PATH` → Invio.
3. Clicca l'icona account in basso a sinistra → *Sign in with GitHub*.
4. Apri la cartella: `File → Open Folder → Documents/final_project_data_science`.

Estensioni consigliate (pannello Estensioni, `Cmd + Shift + X`):
- **GitHub Pull Requests**
- **Python**
- **Jupyter**

## 7. Lavora senza sovrascrivere

Ogni volta che inizi:
1. Source Control → tre puntini → *Pull* (aggiorna `main`).
2. Clicca il nome del branch in basso → *Create new branch* → es. `nome/cosa-fai`.
3. Modifica, poi *Commit*, poi *Publish Branch*.
4. Crea la Pull Request dall'estensione GitHub Pull Requests.
5. Dopo l'approvazione, fai *Pull* su `main`.

Vedi `CONTRIBUTING.md` per le regole complete.

## Problemi comuni

- **`brew: command not found`**: chiudi e riapri il Terminale.
- **`gh: command not found`**: esegui `brew install gh`.
- **`Repository not found`**: non hai ancora accesso, vedi il punto 5.
- **Il comando `code` non funziona**: ripeti il punto 6.1.