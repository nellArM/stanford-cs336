#figure(image("../figures/own.png", width: 100%), caption: [Own estimator (sqrt-MaxRL) vs. its neighbours Dr. GRPO and MaxRL.])

#table(
  columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr), align: (left, center, center, center, center, center, center, center),
  table.header([*method*], [*seeds*], [*val. reward before*], [*final val. reward (mean)*], [*min - max*], [*final format reward*], [*resp. tokens*], [*hours / run*]),
  [dr grpo], [2], [0.004], [*0.365*], [0.284 - 0.446], [0.840], [161], [0.70],
  [maxrl], [2], [0.004], [*0.419*], [0.409 - 0.429], [0.898], [142], [0.65],
  [sqrt maxrl], [2], [0.004], [*0.433*], [0.423 - 0.443], [0.891], [139], [0.70],
)
