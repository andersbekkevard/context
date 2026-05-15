Trene en modell (kanskje forskjellige maskinlæringsmodeller, kanskje nns, kanskje en egen språkmodell) på:
Finn alle sitater fra Benjamin som er "ufaglige". Altså ikke relatert til statistisk læring. Målfunksjon: hvor lættis synes jeg (Anders) det er.

Foreløpig: Enormt datasett (Transcriptions).
- Chunker det til setninger typ. (Må beholde sammenhenger, slik at vi senere kan koble sammen flere setninger til en morsom greie)
- Bruker chat/claude til å lage en del morsomme eksempler, på "hva som er de tingene jeg synes er funny".
- Designe en ergonomisk UI hvor jeg enkelt kan grade setninger, slik at vi får labels på det.
- Bruke de initielle labelene til første trening
- Kontinuerlig symbiose av trening og labeling (jeg sitter bare å spammer grades 1-10 eller 0-1 eller 4 skala eller what?)
- Kan benchmarke flere modeller samtidig (for læringsutbytte)
- Tror setningene må embeddes
- Train/test split må gjøres her. Og CV vil ikke fungere bra (tror jeg?); jo potensielt; fordi er bare rangering, modellen skal aldri "produsere" Benjamin like quotes. Det kan være en naturlig extension.
