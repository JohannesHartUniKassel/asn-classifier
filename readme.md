# ASN Classifier

![Aufbau des Klassifikators](bachelorarbeit.svg)

## Klassifikator:

Der Klassifikator nutzt verscheidene Datenquellen um ASNs zu klassifizieren. Dazu wird PeeringDB als Label genutzt

## Starten:

[`Klassifikator`](notebooks/classificator/classificator6c.ipynb)

Der Klassifikator ist in verschiedenen Notebooks. Dabei gibt es die 6 Klasen und die 10 Klassen Variante.

Zusätzlich dazu sind dort noch Text Late Fusion Modelle, die performen aber schlechter als die mit den einzelwahrscheinlichkeiten direkt ins XGBoost.

## Datenquellen:

Da manche Datenquellen schlecht zu beschaffen sind, habe ich die Dataframes als csv im Repo gelassen. Der Klassifikator sollte also direkt trainierbar sein.