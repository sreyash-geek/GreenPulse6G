**An Advanced simulation environment for a 6G base station with realistic propagation characteristics.**

**State (7-dimensional):**
  1. Network load (0 to 100)
  2. BS mode (0: off, 1: sleep, 2: active)
  3. Average channel quality (0 to 100)
  4. Good channel ratio (fraction of UEs with quality > channel threshold)
  5. Average UE distance from BS (0 to 240)
  6. Average throughput (Mbps, 0 to 50)
  7. Average interference level 

**Propagation Model:**
  - Log-distance path loss: PL = 30 + 10*n*log10(d) + shadowing, with n=3.5, shadowing ~ N(0,8).
  - LOS blockage: if the line from BS to UE intersects an obstacle (building), adding 20 dB loss.
  - Multipath fading: Rayleigh fading loss = 20*log10(r) where r ~ Rayleigh(scale=1).
  - Rain attenuation: if raining, add 5 dB loss.
  - Received power: Pr = Pt - (PL + extra losses). Pt = 46 dBm.
  - Channel quality: mapped linearly from Pr: 0 quality at ≤ -90 dBm, 100 quality at ≥ -60 dBm.

**Throughput:**
  - Computed Throughput = 50 * log2(1 + avg_channel/100). 

**Obstacles:**
  - Two fixed buildings in the environment.

**Actions:**
  - 0: No change, 1: Sleep, 2: Active, 3: Off.

**Reward:**
  - Composite reward that penalizes energy consumption and rewards service quality.

**To run the simulation:**
> python drl_bs.py
