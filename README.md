**An Advanced simulation environment for 6G with realistic propagation characteristics.**

**State (7-dimensional):**
  1. _Network load_ (0 to 100)
  2. _BS mode_ (0: off, 1: sleep, 2: active)
  3. _Average channel quality_ (0 to 100)
  4. _Good channel ratio_ (fraction of UEs with quality > channel threshold)
  5. _Average UE distance from BS_ (0 to 240)
  6. _Average throughput_(Mbps, 0 to 50)
  7. _Average interference level_

**Propagation Model:**
  - _Log-distance path loss:_ PL = 30 + 10*n*log10(d) + shadowing, with n=3.5, shadowing ~ N(0,8).
  - _LOS blockage:_ If the line from BS to UE intersects an obstacle (building), adding 20 dB loss.
  - _Multipath fading:_ Rayleigh fading loss = 20*log10(r) where r ~ Rayleigh(scale=1).
  - _Rain attenuation:_ If raining, add 5 dB loss.
  - _Received power:_ Pr = Pt - (PL + extra losses). Pt = 46 dBm.
  - _Channel quality:_ mapped linearly from Pr: 0 quality at ≤ -90 dBm, 100 quality at ≥ -60 dBm.

**Throughput:**
  - _Computed Throughput_ = 50 * log2(1 + avg_channel/100). 

**Obstacles:**
  - Two fixed buildings in the environment.

**Actions:**
  - 0: No change, 1: Sleep, 2: Active, 3: Off.

**Reward:**
  - Composite reward that _penalizes energy consumption_ and _rewards service quality_.

**To run the simulation:**
> python drl_bs.py
