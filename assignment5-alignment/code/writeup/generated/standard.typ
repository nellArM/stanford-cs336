#figure(image("../figures/standard.png", width: 100%), caption: [Standard on-policy GRPO, r1_zero prompt.])

#table(
  columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr), align: (left, center, center, center, center, center, center, center),
  table.header([*method*], [*seeds*], [*val. reward before*], [*final val. reward (mean)*], [*min - max*], [*final format reward*], [*resp. tokens*], [*hours / run*]),
  [grpo], [2], [0.004], [*0.456*], [0.450 - 0.461], [0.875], [140], [1.04],
)
