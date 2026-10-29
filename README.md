# Dashboard

Per runnare la dashboard bisogna prima di tutto assicurarsi di avere tutte le librerie necessarie installate nel proprio ambiente/venv python:

```bash
pip install -r requirements.txt
```

Dopodiché per avviare la dashboard basta fare:

```bash
panel serve app.py --show --autoreload --websocket-max-message-size 524288000
```

*Nota: il parametro `--websocket-max-message-size 524288000` serve a dare alla dashboard fino a 512MB di RAM, altrimenti crasha con dataset di grandi dimensioni.*