#figure(image("../figures/offpolicy_shuffled.png", width: 100%), caption: [First off-policy implementation (minibatches shuffled across prompts): collapse.])

#table(
  columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr), align: (left, center, center, center, center, center, center, center),
  table.header([*method*], [*seeds*], [*val. reward before*], [*final val. reward (mean)*], [*min - max*], [*final format reward*], [*resp. tokens*], [*hours / run*]),
  [grpo], [2], [0.004], [*0.456*], [0.450 - 0.461], [0.875], [140], [1.04],
  [clip shuffled], [1], [0.004], [*0.000*], [n/a], [0.000], [301], [1.01],
  [gspo shuffled], [1], [0.004], [*0.499*], [n/a], [0.829], [212], [1.23],
  [naive shuffled], [1], [0.004], [*0.000*], [n/a], [0.000], [5], [0.10],
  [noclip shuffled], [1], [0.004], [*0.000*], [n/a], [0.002], [80], [0.57],
)
