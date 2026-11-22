#figure(image("../figures/prompt.png", width: 100%), caption: [Prompt ablation (standard GRPO).])

#table(
  columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr), align: (left, center, center, center, center, center, center, center),
  table.header([*method*], [*seeds*], [*val. reward before*], [*final val. reward (mean)*], [*min - max*], [*final format reward*], [*resp. tokens*], [*hours / run*]),
  [r1 zero], [2], [0.004], [*0.456*], [0.450 - 0.461], [0.875], [140], [1.04],
  [question only], [1], [0.000], [*0.388*], [n/a], [0.952], [367], [1.00],
  [r1 zero three shot], [1], [0.180], [*0.462*], [n/a], [0.927], [115], [1.05],
)
