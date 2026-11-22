#figure(image("../figures/offpolicy.png", width: 100%), caption: [Off-policy variants (32 minibatch steps per rollout batch) vs. on-policy GRPO.])

#table(
  columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr), align: (left, center, center, center, center, center, center, center),
  table.header([*method*], [*seeds*], [*val. reward before*], [*final val. reward (mean)*], [*min - max*], [*final format reward*], [*resp. tokens*], [*hours / run*]),
  [grpo], [2], [0.004], [*0.456*], [0.450 - 0.461], [0.875], [140], [1.04],
  [offpolicy clip], [1], [0.004], [*0.439*], [n/a], [0.961], [133], [0.90],
  [offpolicy gspo], [1], [0.004], [*0.478*], [n/a], [0.935], [154], [0.92],
  [offpolicy naive], [1], [0.004], [*0.450*], [n/a], [0.985], [120], [0.90],
  [offpolicy noclip], [1], [0.004], [*0.505*], [n/a], [0.943], [173], [1.06],
)
