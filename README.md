# Analisi Dati e Dashboard

Script per l'esecuzione dell'analisi dati e l'avvio della dashboard interattiva in Panel.

## Setup

Prima di far partire il codice assicurarsi di avere tutte le librerie necessarie con:

```bash
pip install -r requirements.txt
```

## Utilizzo

1. Esegui l'analisi dati:
   ```bash
   python main.py
   ```

2. Avvia la dashboard:
   ```bash
   panel serve app.py --show --autoreload --websocket-max-message-size 524288000
   ```