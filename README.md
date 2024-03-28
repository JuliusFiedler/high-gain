# Workflow
- in `system.py` Ausgang und Beobachterdimension entsprechend einstellen.
- `train.py` trainiert NN
    - entweder als ein Netz oder aufgeteilt nach q^-1 und alpha
- `main.py` für simulation
    - erstellt ne reihe von plots

# Notes
## inv pendulum
- mit noise funktiniert nicht
- N4 x3, alpha=0, nur sehr wenig schlechter als trainiertes alpha
## double pendulum
- in polarkoord. (4 states) funktioniert für kleine Auslenkungen
- polar: alpha=0, nur sehr wenig schlechter als trainiertes alpha
- kartesische koord (6 states) funktioniert nix
## magnetic pendulum
- embedded observer gut, reconstruction in x schlecht