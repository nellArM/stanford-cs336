#table(columns: 5, align: (left, right, right, right, right), stroke: none, inset: (x: 5pt, y: 3pt),
  table.hline(),
  table.header([*context*], [*forward fp32 (GiB)*], [*forward bf16 (GiB)*], [*train step fp32 (GiB)*], [*train step bf16 (GiB)*]),
  table.hline(stroke: 0.5pt),
  [128], [12.9], [19.1], [OOM], [OOM],
  [2,048], [21.3], [25.4], [OOM], [OOM],
  table.hline(),
)
