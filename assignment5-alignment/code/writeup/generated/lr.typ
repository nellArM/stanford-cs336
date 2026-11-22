#figure(image("../figures/lr.png", width: 100%), caption: [Learning-rate sweep (standard GRPO).])

_Runs without a final evaluation (they stopped before step 200; see the text): grpo lr3e-05 s0._

#figure(image("../figures/lr_final.png", width: 55%), caption: [Final validation reward versus learning rate.])

#table(
  columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr), align: (left, center, center, center, center, center, center, center),
  table.header([*method*], [*seeds*], [*val. reward before*], [*final val. reward (mean)*], [*min - max*], [*final format reward*], [*resp. tokens*], [*hours / run*]),
  [lr=3e-06], [1], [0.004], [*0.149*], [n/a], [0.788], [108], [0.71],
  [lr=1e-05], [2], [0.004], [*0.456*], [0.450 - 0.461], [0.875], [140], [1.04],
)
