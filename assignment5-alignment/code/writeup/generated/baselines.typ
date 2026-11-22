#table(
  columns: (auto, auto, 1fr, 1fr, 1fr, auto), align: (left, center, center, center, center, center),
  table.header([*prompt*], [*n*], [*(1) format 1, answer 1*], [*(2) format 1, answer 0*], [*(3) format 0, answer 0*], [*mean resp. tokens*]),
  [`question_only`], [1319], [1 (0.1%)], [308 (23.4%)], [1010 (76.6%)], [259],
  [`r1_zero`], [1319], [1 (0.1%)], [654 (49.6%)], [664 (50.3%)], [109],
  [`r1_zero_three_shot`], [1319], [254 (19.3%)], [997 (75.6%)], [68 (5.2%)], [108],
)
