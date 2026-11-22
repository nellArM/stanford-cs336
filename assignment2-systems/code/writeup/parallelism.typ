#import "lib.typ": gen

// Section 8 of the handout: analysing parallelism strategies (pure pen-and-paper problems).

= Analysing parallelism strategies <sec-parallelism>

Notation as in the handout: $N$ devices, each with egress bandwidth $W$ (bytes/s) and speed $C$ (FLOP/s); the FFN has input $bold(x)$ of shape $(B, D)$, weights $bold(W)_1, bold(W)_2$ of shape $(D, D_"FF")$ and $bold(W)_3$ of shape $(D_"FF", D)$; everything is FP16 (2 bytes per element); a matmul $(A,B)(B,C)$ costs $2 A B C$ FLOPs.

== Problem `alternate_ring_all_reduce`

The algorithm takes
$ T = (N - 1) S / W $
seconds: in each of the $N - 1$ steps every device forwards one *whole* tensor of $S$ bytes (not a chunk of $S\/N$), which takes $S\/W$ at egress bandwidth $W$. This is $N\/2$ times slower than reduce-scatter + all-gather, which needs $2 (N-1)/N dot S/W$.

== Problem `data_parallel_calcs`

*(a) Backward FLOPs.* Each device runs the backward pass on $B\/N_"DP"$ rows. It contains six matmuls of size $(B\/N_"DP") times D times D_"FF"$: $bold(d z) = bold(d y) bold(W)_3^top$, the two products in $bold(d x) = bold(d x)_1 bold(W)_1^top + bold(d x)_2 bold(W)_2^top$, and the three weight gradients $bold(z)^top bold(d y)$, $bold(x)^top bold(d x)_2$, $bold(x)^top bold(d x)_1$. Hence
$ F_"bwd" = 6 dot 2 dot B / N_"DP" dot D dot D_"FF" = (12 B D D_"FF") / N_"DP" quad "FLOPs per device" $
(the total over all devices is still $12 B D D_"FF"$; data parallelism only divides the work).

*(b) Backward communication time.* The three weight gradients ($3 D D_"FF"$ elements $= 6 D D_"FF"$ bytes) must be all-reduced; a ring all-reduce of $S$ bytes takes $2 (N-1)/N dot S/W$, so
$ T_"comm" = (12 D D_"FF") / W dot (N_"DP" - 1) / N_"DP". $
It does not depend on $B$: the gradients have the size of the weights.

*(c) Limit on $N_"DP"$.* We stay compute-bound while $T_"comm" <= F_"bwd" \/ C$:
$ (12 D D_"FF") / W dot (N_"DP" - 1) / N_"DP" <= (12 B D D_"FF") / (N_"DP" C)
  quad <==> quad N_"DP" <= 1 + (B W) / C. $
Adding devices shrinks each device's compute ($prop 1\/N_"DP"$) while the all-reduce cost stays roughly constant, so only a larger batch (or faster links relative to the accelerator) buys more devices; the model size cancels.

== Problem `fsdp_calcs`

*(a) FLOPs.* The compute is exactly the data-parallel compute on $B\/N_"FSDP"$ rows: the backward pass has the same six matmuls, $F_"bwd" = 12 B D D_"FF" \/ N_"FSDP"$, and the forward pass has three ($bold(x) bold(W)_1$, $bold(x) bold(W)_2$, $bold(z) bold(W)_3$), $F_"fwd" = 6 B D D_"FF" \/ N_"FSDP"$ per device.

*(b) Communication time.* Forward: three all-gathers of $S = 2 D D_"FF"$ bytes each, a ring all-gather taking $(N-1)/N dot S/W$:
$ T_"fwd" = (6 D D_"FF") / W dot (N_"FSDP" - 1) / N_"FSDP". $
Backward: the same three all-gathers plus three reduce-scatters of the weight gradients (same size, same cost each):
$ T_"bwd" = (12 D D_"FF") / W dot (N_"FSDP" - 1) / N_"FSDP", $
i.e. exactly the cost of data parallelism's all-reduce (an all-reduce _is_ a reduce-scatter plus an all-gather; FSDP just gathers weights instead of gradients).

*(c) Limits.* Backward: $T_"bwd" <= F_"bwd"\/C$ gives $N_"FSDP" <= 1 + B W\/C$. Forward: $T_"fwd" <= F_"fwd"\/C$, i.e. $6 D D_"FF" (N-1) \/ (N W) <= 6 B D D_"FF" \/ (N C)$, gives the same bound $N_"FSDP" <= 1 + B W\/C$. Both passes have a 2:1 ratio in compute and in communication, so FSDP scales exactly as far as plain DP — the memory savings are free as long as the collectives are overlapped with compute.

== Problem `tp_calcs`

*(a) Backward pass.* $bold(d y)$ (shape $(B, D)$) is available on every device, because $bold(y)$ was produced by an all-reduce. Device $i$ computes
$ bold(d z)^((i)) &= bold(d y) thin bold(W)_3^((i) top) && quad (B, D_"FF" \/ N_"TP") \
  bold(d x)_2^((i)) &= bold(d z)^((i)) * f(bold(x)_1^((i))) \
  bold(d x)_1^((i)) &= bold(d z)^((i)) * f'(bold(x)_1^((i))) * bold(x)_2^((i)) \
  bold(d W)_3^((i)) &= bold(z)^((i) top) bold(d y) && quad (D_"FF" \/ N_"TP", D) \
  bold(d W)_2^((i)) &= bold(x)^top bold(d x)_2^((i)) && quad (D, D_"FF" \/ N_"TP") \
  bold(d W)_1^((i)) &= bold(x)^top bold(d x)_1^((i)) \
  bold(d x)^((i)) &= bold(d x)_1^((i)) bold(W)_1^((i) top) + bold(d x)_2^((i)) bold(W)_2^((i) top) && quad (B, D) "partial sum" \
  bold(d x) &= "all-reduce"({bold(d x)^((i))}_(i=0)^(N_"TP" - 1)). $
The weight gradients need no communication (each device owns its shard); the only collective is the all-reduce of the input gradient, mirroring the all-reduce of $bold(y)$ in the forward pass.

*(b) FLOPs per device.* Every matmul now has one dimension divided by $N_"TP"$ (the batch is _not_ sharded). Forward: three matmuls of $2 B D D_"FF"\/N_"TP"$, so $F_"fwd" = 6 B D D_"FF" \/ N_"TP"$. Backward: six such matmuls, $F_"bwd" = 12 B D D_"FF" \/ N_"TP"$.

*(c) Communication time.* Forward: one all-reduce of $bold(y)$, $S = 2 B D$ bytes: $T_"fwd" = (4 B D)/W dot (N_"TP" - 1)/N_"TP"$. Backward: one all-reduce of $bold(d x)$, also $2 B D$ bytes: $T_"bwd" = (4 B D)/W dot (N_"TP" - 1)/N_"TP"$. Unlike DP/FSDP this grows with the _activations_ ($B D$) rather than with the weights.

*(d) Limits.* Forward: $(4 B D)/W dot (N_"TP"-1)/N_"TP" <= (6 B D D_"FF")/(N_"TP" C)$ gives
$ N_"TP" <= 1 + (3 D_"FF" W) / (2 C). $
Backward: the compute doubles while the communication does not, so
$ N_"TP" <= 1 + (3 D_"FF" W) / C. $
The batch size cancels: TP is limited by the layer width $D_"FF"$, DP/FSDP by the batch size — which is why the two compose well.

== Problem `fsdp_tp_calcs`

*(a) Forward FLOPs.* Each device sees $B\/N_"FSDP"$ rows and $1\/N_"TP"$ of each weight matrix: three matmuls of $2 dot B/N_"FSDP" dot D dot D_"FF"/N_"TP"$, so
$ F_"fwd" = (6 B D D_"FF") / (N_"FSDP" N_"TP") quad "per device". $

*(b) Forward communication.* FSDP axis: three all-gathers over $N_"FSDP"$ devices of the TP-sharded weights, each of $2 D D_"FF" \/ N_"TP"$ bytes. TP axis: one all-reduce over $N_"TP"$ devices of $bold(y)^((i,j))$, $2 B D \/ N_"FSDP"$ bytes. Because the two axes overlap, the time is the larger of the two:
$ T_"comm" = max( underbrace((6 D D_"FF") / (N_"TP" W) dot (N_"FSDP" - 1) / N_"FSDP", "FSDP all-gathers"), quad
               underbrace((4 B D) / (N_"FSDP" W) dot (N_"TP" - 1) / N_"TP", "TP all-reduce") ). $

*(c) Largest $N$ with overlapping axes.* Requiring each term to be at most $F_"fwd" \/ C = 6 B D D_"FF" \/ (N_"FSDP" N_"TP" C)$:
$ "FSDP:" quad & (6 D D_"FF" (N_"FSDP" - 1)) / (N_"TP" N_"FSDP" W) <= (6 B D D_"FF") / (N_"FSDP" N_"TP" C) quad <==> quad N_"FSDP" <= 1 + (B W) / C, \
  "TP:" quad & (4 B D (N_"TP" - 1)) / (N_"FSDP" N_"TP" W) <= (6 B D D_"FF") / (N_"FSDP" N_"TP" C) quad <==> quad N_"TP" <= 1 + (3 D_"FF" W) / (2 C). $
The two conditions decouple — each is the single-axis bound from before — so both can be saturated at once:
$ N = N_"TP" N_"FSDP" <= (1 + (B W) / C) (1 + (3 D_"FF" W) / (2 C)). $
2D parallelism multiplies the two limits because sharding along one axis shrinks the other axis's payload by the same factor as it shrinks the compute.

*(d) Largest $N$ when the axes share the network.* Now the two costs add:
$ (6 D D_"FF" (N_"FSDP" - 1)) / (N_"TP" N_"FSDP" W) + (4 B D (N_"TP" - 1)) / (N_"FSDP" N_"TP" W) <= (6 B D D_"FF") / (N_"FSDP" N_"TP" C). $
Multiplying by $N_"FSDP" N_"TP" W \/ (2 D)$:
$ 3 D_"FF" (N_"FSDP" - 1) + 2 B (N_"TP" - 1) <= (3 B D_"FF" W) / C
  quad <==> quad 3 D_"FF" N_"FSDP" + 2 B N_"TP" <= K, quad K := (3 B D_"FF" W) / C + 3 D_"FF" + 2 B. $
We maximise the product $N_"FSDP" N_"TP"$ under a linear budget; by AM–GM (or a Lagrange multiplier) the optimum spends half of the budget on each term, $3 D_"FF" N_"FSDP" = 2 B N_"TP" = K\/2$, i.e. $N_"FSDP" = K \/ (6 D_"FF")$ and $N_"TP" = K \/ (4 B)$. Therefore
$ N = N_"TP" N_"FSDP" <= K^2 / (24 B D_"FF") = ((3 B D_"FF" W) / C + 3 D_"FF" + 2 B)^2 / (24 B D_"FF"). $
For fast links ($B W \/ C >> 1$ and $D_"FF" W\/C >> 1$) this is $approx 3/8 dot B D_"FF" (W\/C)^2$, exactly a quarter of the overlapped limit $approx 3/2 dot B D_"FF" (W\/C)^2$: each axis may only use half of the communication budget, which halves the device count along both axes.
