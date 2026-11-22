#figure(image("../figures/variants.png", width: 100%), caption: [On-policy estimator variants.])

#table(
  columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr), align: (left, center, center, center, center, center, center, center),
  table.header([*method*], [*seeds*], [*val. reward before*], [*final val. reward (mean)*], [*min - max*], [*final format reward*], [*resp. tokens*], [*hours / run*]),
  [grpo], [2], [0.004], [*0.456*], [0.450 - 0.461], [0.875], [140], [1.04],
  [dr grpo], [2], [0.004], [*0.365*], [0.284 - 0.446], [0.840], [161], [0.70],
  [grpo constant], [2], [0.004], [*0.404*], [0.368 - 0.440], [0.881], [145], [0.83],
  [maxrl], [2], [0.004], [*0.419*], [0.409 - 0.429], [0.898], [142], [0.65],
  [rft], [2], [0.004], [*0.427*], [0.423 - 0.432], [0.957], [158], [0.62],
)
