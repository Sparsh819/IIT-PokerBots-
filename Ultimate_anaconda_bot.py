'''
Ultimate Anaconda Hold'em Bot — IIT Pokerbots 2026 Finals
═════════════════════════════════════════════════════════════
Three-pillar strategy:
  1. Perfect-information endgame via card tracking (know 6-7 of opponent's cards)
  2. Dual-objective card passing (maximize own hand + sabotage opponent)
  3. GTO-baseline betting with exploitative drift over 1000 rounds
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionPass
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

from collections import Counter
from itertools import combinations
import random

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
RANK_CHAR = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
             'T':10,'J':11,'Q':12,'K':13,'A':14}
SUIT_MAP  = {'d':0,'h':1,'s':2,'c':3}
FULL_DECK = [r+s for r in '23456789TJQKA' for s in 'dhsc']

# Pre-computed combination index tuples
_C75 = ((0,1,2,3,4),(0,1,2,3,5),(0,1,2,3,6),(0,1,2,4,5),(0,1,2,4,6),
        (0,1,2,5,6),(0,1,3,4,5),(0,1,3,4,6),(0,1,3,5,6),(0,1,4,5,6),
        (0,2,3,4,5),(0,2,3,4,6),(0,2,3,5,6),(0,2,4,5,6),(0,3,4,5,6),
        (1,2,3,4,5),(1,2,3,4,6),(1,2,3,5,6),(1,2,4,5,6),(1,3,4,5,6),
        (2,3,4,5,6))
_C73 = tuple(combinations(range(7), 3))
_C72 = tuple(combinations(range(7), 2))


# ══════════════════════════════════════════════════════════════
# HAND EVALUATION — optimised pure-Python, ~75K evals/sec
# Tuple-based comparison: higher = better hand
# Fixed 6-element tuples for uniform fast comparison
# ══════════════════════════════════════════════════════════════
def _e5(r0, r1, r2, r3, r4, fl):
    """Evaluate 5 ranks (pre-sorted desc) + flush flag → 6-tuple."""
    # Straight
    is_str = False
    hi = r0
    if r0 != r1 and r1 != r2 and r2 != r3 and r3 != r4:
        if r0 - r4 == 4:
            is_str = True
        elif r0 == 14 and r1 == 5 and r4 == 2:
            is_str = True
            hi = 5
    if is_str and fl:
        return (8, hi, 0, 0, 0, 0)
    # Quads
    if r0 == r1 == r2 == r3:
        return (7, r0, r4, 0, 0, 0)
    if r1 == r2 == r3 == r4:
        return (7, r1, r0, 0, 0, 0)
    # Full house
    if r0 == r1 == r2 and r3 == r4:
        return (6, r0, r3, 0, 0, 0)
    if r0 == r1 and r2 == r3 == r4:
        return (6, r2, r0, 0, 0, 0)
    # Flush
    if fl:
        return (5, r0, r1, r2, r3, r4)
    # Straight
    if is_str:
        return (4, hi, 0, 0, 0, 0)
    # Trips
    if r0 == r1 == r2:
        return (3, r0, r3, r4, 0, 0)
    if r1 == r2 == r3:
        return (3, r1, r0, r4, 0, 0)
    if r2 == r3 == r4:
        return (3, r2, r0, r1, 0, 0)
    # Two pair
    if r0 == r1 and r2 == r3:
        return (2, r0, r2, r4, 0, 0)
    if r0 == r1 and r3 == r4:
        return (2, r0, r3, r2, 0, 0)
    if r1 == r2 and r3 == r4:
        return (2, r1, r3, r0, 0, 0)
    # One pair
    if r0 == r1:
        return (1, r0, r2, r3, r4, 0)
    if r1 == r2:
        return (1, r1, r0, r3, r4, 0)
    if r2 == r3:
        return (1, r2, r0, r1, r4, 0)
    if r3 == r4:
        return (1, r3, r0, r1, r2, 0)
    # High card
    return (0, r0, r1, r2, r3, r4)


def eval7(cards_str):
    """Best 5-from-7 card strings → comparable 6-tuple."""
    R = [RANK_CHAR[c[0]] for c in cards_str]
    S = [SUIT_MAP[c[1]] for c in cards_str]
    best = (0, 0, 0, 0, 0, 0)
    for a, b, c, d, e in _C75:
        rs = sorted((R[a], R[b], R[c], R[d], R[e]), reverse=True)
        fl = S[a] == S[b] == S[c] == S[d] == S[e]
        sc = _e5(rs[0], rs[1], rs[2], rs[3], rs[4], fl)
        if sc > best:
            best = sc
    return best


def eval_n(cards_str):
    """Best 5-from-N (N >= 5)."""
    R = [RANK_CHAR[c[0]] for c in cards_str]
    S = [SUIT_MAP[c[1]] for c in cards_str]
    n = len(R)
    best = (0, 0, 0, 0, 0, 0)
    for combo in combinations(range(n), 5):
        rs = sorted((R[i] for i in combo), reverse=True)
        fl = len({S[i] for i in combo}) == 1
        sc = _e5(rs[0], rs[1], rs[2], rs[3], rs[4], fl)
        if sc > best:
            best = sc
    return best


# ══════════════════════════════════════════════════════════════
# EQUITY ESTIMATION
# ══════════════════════════════════════════════════════════════
_EQ_BASE   = {8:.995, 7:.985, 6:.935, 5:.840, 4:.775, 3:.670, 2:.510, 1:.320, 0:.130}
_EQ_SPREAD = {8:.004, 7:.010, 6:.040, 5:.060, 4:.040, 3:.070, 2:.090, 1:.110, 0:.070}

def equity_heuristic(rank_tuple):
    """O(1) equity estimate from hand rank tuple."""
    cat = rank_tuple[0]
    base = _EQ_BASE[cat]
    sprd = _EQ_SPREAD[cat]
    k = (rank_tuple[1] - 2) / 12.0
    return base - sprd + 2 * sprd * k


# ══════════════════════════════════════════════════════════════
# CARD-PASS SCORING HEURISTICS
# ══════════════════════════════════════════════════════════════
def _keep_score(ranks, suits, n_keep):
    """Score a kept-card set by hand-building potential."""
    sc = 0.0
    cnt = Counter(ranks)
    scnt = Counter(suits)
    cv = sorted(cnt.values(), reverse=True)

    # Made-hand components
    if cv[0] == 4:
        sc += 12000
    elif cv[0] == 3:
        sc += 8500 if (len(cv) > 1 and cv[1] >= 2) else 6000
    elif cv[0] == 2:
        np_ = sum(1 for v in cv if v == 2)
        sc += 3500 if np_ >= 2 else 1800
        sc += max(r for r, c_ in cnt.items() if c_ >= 2) * 30

    # Flush potential
    mx = max(scnt.values())
    if n_keep <= 4:
        sc += 3500 if mx == 4 else (900 if mx == 3 else 0)
    elif n_keep == 5:
        sc += 5500 if mx == 5 else (1500 if mx == 4 else 0)
    else:
        sc += 5500 if mx >= 5 else (800 if mx == 4 else 0)

    # Straight potential
    uq = set(ranks)
    if 14 in uq:
        uq.add(1)
    best_st = 0
    for base in range(1, 11):
        have = len(uq & set(range(base, base + 5)))
        if have == 5:
            best_st = max(best_st, 1400)
        elif have == 4:
            best_st = max(best_st, 700)
        elif have == 3 and n_keep <= 5:
            best_st = max(best_st, 250)
    sc += best_st

    # High-card value
    sc += sum(ranks) * 4
    return sc


def _pass_danger(ranks, suits):
    """Penalty for how helpful passed cards are to opponent."""
    pen = 0.0
    scnt = Counter(suits)
    for c_ in scnt.values():
        if c_ >= 3:
            pen += 280
        elif c_ == 2:
            pen += 110
    sr = sorted(ranks)
    for i in range(len(sr)):
        for j in range(i + 1, len(sr)):
            gap = sr[j] - sr[i]
            if gap == 0:
                pen += 400
            elif gap <= 2:
                pen += 75
            elif gap <= 4:
                pen += 35
    for r in ranks:
        if r == 14:
            pen += 50
        elif r >= 12:
            pen += 30
    return pen


# ══════════════════════════════════════════════════════════════
# THE BOT
# ══════════════════════════════════════════════════════════════
class Player(BaseBot):

    def __init__(self) -> None:
        # Persistent opponent model
        self.rounds_played     = 0
        self.opp_folds         = 0
        self.opp_raises        = 0
        self.opp_total_acts    = 0
        # Per-round state
        self._reset_round()

    def _reset_round(self):
        self.cards_seen     = set()
        self.passed_to_opp  = set()
        self.prev_street    = None

    # ──────────────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────────────
    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self._reset_round()
        self.cards_seen = set(current_state.my_hand)

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.rounds_played += 1
        opp = current_state.opp_hand
        payoff = current_state.payoff
        # If we won without showdown → opponent folded
        if (not opp or len(opp) == 0) and payoff > 0:
            self.opp_folds += 1
            self.opp_total_acts += 1

    # ──────────────────────────────────────────────────────
    # OPPONENT TRACKING (lightweight, inline)
    # ──────────────────────────────────────────────────────
    def _track(self, state):
        street = state.street
        if street and street.startswith('Betting'):
            ctc = state.cost_to_call
            if ctc > 0 and street != self.prev_street:
                self.opp_raises += 1
                self.opp_total_acts += 1
            self.prev_street = street

    def _record_opp_fold(self):
        self.opp_folds += 1
        self.opp_total_acts += 1

    @property
    def opp_fold_rate(self):
        return self.opp_folds / self.opp_total_acts if self.opp_total_acts > 30 else 0.33

    # ──────────────────────────────────────────────────────
    # CARD TRACKING
    # ──────────────────────────────────────────────────────
    def _update_tracking(self, my_hand):
        self.cards_seen.update(my_hand)

    def _deduce_opp(self, my_hand):
        known_opp = self.cards_seen - set(my_hand)
        return known_opp, max(7 - len(known_opp), 0)

    # ──────────────────────────────────────────────────────
    # EXACT EQUITY (used in later betting rounds)
    # ──────────────────────────────────────────────────────
    def _exact_equity(self, my_hand, known_opp, n_unknown):
        my_rank = eval7(my_hand)
        ko = list(known_opp)

        if n_unknown == 0:
            opp_rank = eval7(ko)
            return 1.0 if my_rank > opp_rank else (0.5 if my_rank == opp_rank else 0.0)

        remaining = [c for c in FULL_DECK if c not in self.cards_seen]

        if n_unknown == 1:
            wins = ties = 0
            for card in remaining:
                opp_rank = eval7(ko + [card])
                if my_rank > opp_rank:
                    wins += 1
                elif my_rank == opp_rank:
                    ties += 1
            t = len(remaining)
            return (wins + 0.5 * ties) / t if t else 0.5

        if n_unknown == 2:
            combos = list(combinations(remaining, 2))
            if len(combos) > 100:
                combos = random.sample(combos, 100)
            wins = ties = 0
            for c1, c2 in combos:
                opp_rank = eval7(ko + [c1, c2])
                if my_rank > opp_rank:
                    wins += 1
                elif my_rank == opp_rank:
                    ties += 1
            t = len(combos)
            return (wins + 0.5 * ties) / t if t else 0.5

        return equity_heuristic(my_rank)

    # ──────────────────────────────────────────────────────
    # CARD PASSING
    # ──────────────────────────────────────────────────────
    def _pass_cards(self, my_hand, n_pass):
        n = len(my_hand)
        n_keep = n - n_pass
        R = [RANK_CHAR[c[0]] for c in my_hand]
        S = [c[1] for c in my_hand]

        if n_pass == 3:
            combos = _C73
        elif n_pass == 2:
            combos = _C72
        else:
            combos = tuple((i,) for i in range(n))

        best_sc = -1e18
        best_pi = [0]

        for pi in combos:
            pi_set = set(pi)
            ki = [i for i in range(n) if i not in pi_set]
            kr = [R[i] for i in ki]
            ks = [S[i] for i in ki]
            pr = [R[i] for i in pi]
            ps = [S[i] for i in pi]

            if n_keep >= 5:
                kc = [my_hand[i] for i in ki]
                hr = eval_n(kc)
                kscore = hr[0] * 1e10
                for idx, v in enumerate(hr[1:], 1):
                    kscore += v * (100 ** (5 - idx))
            else:
                kscore = _keep_score(kr, ks, n_keep)

            danger = _pass_danger(pr, ps)
            alpha = 0.9 if n_pass == 3 else (0.7 if n_pass == 2 else 0.5)
            total = kscore - alpha * danger

            if total > best_sc:
                best_sc = total
                best_pi = list(pi)

        for i in best_pi:
            self.passed_to_opp.add(my_hand[i])
        return sorted(best_pi)

    def _single_pass_blocking(self, my_hand):
        """SinglePass with blocking: avoid giving opponent their flush/straight card."""
        n = len(my_hand)
        R = [RANK_CHAR[c[0]] for c in my_hand]
        S = [c[1] for c in my_hand]

        known_opp, _ = self._deduce_opp(my_hand)
        flush_suit = None
        if len(known_opp) >= 4:
            opp_scnt = Counter(c[1] for c in known_opp)
            for s, cnt in opp_scnt.items():
                if cnt >= 4:
                    flush_suit = s
                    break

        # Opponent's rank profile for straight blocking
        opp_ranks = set()
        if len(known_opp) >= 4:
            opp_ranks = {RANK_CHAR[c[0]] for c in known_opp}
            if 14 in opp_ranks:
                opp_ranks.add(1)

        best_sc = -1e18
        best_i = 0

        for i in range(n):
            ki = [j for j in range(n) if j != i]
            kc = [my_hand[j] for j in ki]
            hr = eval_n(kc)
            kscore = hr[0] * 1e10
            for idx, v in enumerate(hr[1:], 1):
                kscore += v * (100 ** (5 - idx))

            danger = _pass_danger([R[i]], [S[i]])

            # Blocking: penalty for completing opponent's flush
            if flush_suit and S[i] == flush_suit:
                danger += 250

            # Blocking: penalty for completing opponent's straight
            if opp_ranks:
                cr = R[i]
                for base in range(max(1, cr - 4), min(11, cr + 1)):
                    window = set(range(base, base + 5))
                    if cr in window and len((window - {cr}) & opp_ranks) >= 3:
                        danger += 120
                        break

            total = kscore - 0.5 * danger
            if total > best_sc:
                best_sc = total
                best_i = i

        self.passed_to_opp.add(my_hand[best_i])
        return [best_i]

    # ──────────────────────────────────────────────────────
    # BETTING ENGINE
    # ──────────────────────────────────────────────────────
    def _bet(self, state, equity, perfect_info=False):
        pot     = state.pot
        ctc     = state.cost_to_call
        facing  = ctc > 0
        pot_odds = ctc / (pot + ctc) if ctc > 0 else 0.0
        fold_r  = self.opp_fold_rate
        exploit = self.rounds_played >= 80

        def _raise(frac):
            if not state.can_act(ActionRaise):
                return None
            mn, mx = state.raise_bounds
            amt = int(pot * frac) + state.opp_wager
            amt = max(mn, min(amt, mx))
            return ActionRaise(amt)

        def _safe(action):
            """Return action or safest fallback."""
            if action:
                return action
            if state.can_act(ActionCall):
                return ActionCall()
            if state.can_act(ActionCheck):
                return ActionCheck()
            return ActionFold()

        # ── Perfect info: play tighter, more aggressive with winners ──
        if perfect_info:
            if facing:
                if equity >= 0.92:
                    return _safe(_raise(1.0))
                if equity >= 0.70:
                    return _safe(_raise(0.65))
                if equity >= 0.50:
                    return ActionCall() if state.can_act(ActionCall) else ActionCheck()
                if equity >= pot_odds:
                    return ActionCall() if state.can_act(ActionCall) else ActionCheck()
                return ActionFold() if state.can_act(ActionFold) else ActionCall()
            else:
                if equity >= 0.90:
                    return _safe(_raise(0.75))
                if equity >= 0.65:
                    return _safe(_raise(0.50))
                if equity >= 0.50:
                    return _safe(_raise(0.35))
                return ActionCheck() if state.can_act(ActionCheck) else ActionCall()

        # ── Standard GTO + exploit ──
        if facing:
            if equity >= 0.85:
                return _safe(_raise(0.80))
            if equity >= 0.68:
                if random.random() < 0.35:
                    return _safe(_raise(0.60))
                return ActionCall() if state.can_act(ActionCall) else ActionCheck()
            if equity >= 0.50:
                return ActionCall() if state.can_act(ActionCall) else ActionCheck()
            if equity >= pot_odds:
                return ActionCall() if state.can_act(ActionCall) else ActionCheck()
            # Below pot odds
            if exploit and fold_r > 0.55 and equity > 0.10 and random.random() < 0.18:
                r = _raise(0.85)
                if r:
                    return r
            return ActionFold() if state.can_act(ActionFold) else ActionCall()
        else:
            if equity >= 0.80:
                return _safe(_raise(0.65))
            if equity >= 0.60:
                return _safe(_raise(0.45))
            if equity >= 0.45:
                if random.random() < 0.30:
                    return _safe(_raise(0.35))
                return ActionCheck() if state.can_act(ActionCheck) else ActionCall()
            # Weak: bluff or check
            if exploit and equity < 0.22 and fold_r > 0.48 and random.random() < 0.22:
                r = _raise(0.55)
                if r:
                    return r
            return ActionCheck() if state.can_act(ActionCheck) else ActionCall()

    # ══════════════════════════════════════════════════════
    # MAIN DECISION
    # ══════════════════════════════════════════════════════
    def get_move(self, game_info: GameInfo, current_state: PokerState):
        my_hand  = list(current_state.my_hand)
        street   = current_state.street
        t_left   = game_info.time_bank
        emergency = t_left < 2.0

        self._update_tracking(my_hand)
        self._track(current_state)

        # ── CARD PASSING ──
        if street in ('TriplePass', 'DoublePass', 'SinglePass'):
            n_pass = {'TriplePass': 3, 'DoublePass': 2, 'SinglePass': 1}[street]

            if emergency:
                idx = sorted(range(len(my_hand)), key=lambda i: RANK_CHAR[my_hand[i][0]])
                pi = sorted(idx[:n_pass])
                for i in pi:
                    self.passed_to_opp.add(my_hand[i])
                return ActionPass(pi)

            if street == 'SinglePass':
                return ActionPass(self._single_pass_blocking(my_hand))
            return ActionPass(self._pass_cards(my_hand, n_pass))

        # ── BETTING ──
        if street == 'Betting#3' and not emergency:
            ko, nu = self._deduce_opp(my_hand)
            if len(ko) >= 5:
                eq = self._exact_equity(my_hand, ko, nu)
                return self._bet(current_state, eq, perfect_info=True)

        if street == 'Betting#2' and not emergency:
            ko, nu = self._deduce_opp(my_hand)
            if len(ko) >= 3 and nu <= 4:
                eq = self._exact_equity(my_hand, ko, nu)
                return self._bet(current_state, eq)

        # Heuristic equity for Betting#1 or emergencies
        rank = eval7(my_hand)
        eq   = equity_heuristic(rank)
        return self._bet(current_state, eq)


if __name__ == '__main__':
    run_bot(Player(), parse_args())