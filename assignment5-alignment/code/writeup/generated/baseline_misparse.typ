#table(
  columns: (auto, auto, auto, 1fr, 1fr), align: (left, left, center, center, center),
  table.header([*prompt*], [*category*], [*n*], [*last number in the response equals the ground truth*], [*hit the 512-token limit*]),
  [`question_only`], [(2) formatted, graded wrong], [308], [6], [128],
  [`question_only`], [(3) unformatted], [1010], [20], [306],
  [`r1_zero`], [(2) formatted, graded wrong], [654], [57], [0],
  [`r1_zero`], [(3) unformatted], [664], [47], [36],
  [`r1_zero_three_shot`], [(2) formatted, graded wrong], [997], [16], [0],
  [`r1_zero_three_shot`], [(3) unformatted], [68], [3], [8],
)
