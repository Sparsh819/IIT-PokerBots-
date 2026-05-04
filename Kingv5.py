"""
KING v5 — Surgical Upgrade over v3
====================================
v3's proven betting + trap logic preserved EXACTLY.

Changes (each strictly better):
  1. Exhaustive pass for DoublePass (C(7,2)=21 combos vs greedy — exact scoring)
  2. Anti-synergy: mild penalty for feeding opponent pairs/flushes on DoublePass/SinglePass  
  3. Improved card tracker (all-seen) → more known opponent cards → better equity
  4. Wider binary overrides: 0.12/0.88 (was 0.08/0.92) — more hands correctly folded/valued
  5. Equity computed on ALL betting streets (returns None when infeasible)

ALL betting logic below the binary overrides is IDENTICAL to v3 (the #1 bot).
"""