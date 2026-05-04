"""
KING v4 — Tournament-Dominant Anaconda Hold'em Bot
===================================================
Built on bot_pass_trap's proven foundation with surgical upgrades:

  BUGS FIXED (from v3):
  • eval7.evaluate() called on 7 cards (expects 5) → now uses best_5_of_n
  • Card tracker only tracked cards_given → now tracks ALL seen cards

  STRATEGIC UPGRADES:
  1. Exhaustive pass enumeration (C(7,3)=35) vs greedy removal
  2. Anti-synergy: avoids passing cards that strengthen opponent's known hand
  3. All-seen card tracking → know 6-7 of opponent's 7 cards by Betting#3
  4. Exact equity used for smooth betting decisions, not just binary overrides
  5. Time-aware: skips heavy computation when time bank is low
"""

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionPass
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
from itertools import combinations
from collections import Counter

# ─── Constants ────────────────────────────────────────────────────────────────

RANK_STR  = '23456789TJQKA'
SUITS     = 'cdhs'
FULL_DECK = [r + s for r in RANK_STR for s in SUITS]
E7        = {c: eval7.Card(c) for c in FULL_DECK}

MAX_SCORE   = 135_004_167
PASS_COUNTS = {'TriplePass': 3, 'DoublePass': 2, 'SinglePass': 1}


# ─── Core Hand Evaluation ────────────────────────────────────────────────────

def best_5_of_n(cards):
    """Evaluate the best 5-card hand from n cards."""
    if len(cards) == 5:
        return eval7.evaluate(cards)
    return max(eval7.evaluate(list(c)) for c in combinations(cards, 5))


def partial_score(cards):
    """Heuristic score for < 5 cards (used during pass selection)."""
    if not cards:
        return 0
    ranks = sorted([RANK_STR.index(str(c)[0]) for c in cards], reverse=True)
    suits = [str(c)[1] for c in cards]
    rc, sc = Counter(ranks), Counter(suits)
    counts = sorted(rc.values(), reverse=True)

    score = sum(r * 5_000 for r in ranks)
    if   counts[0] >= 4:                                          score += 80_000_000
    elif counts[0] == 3:                                          score += 25_000_000
    elif counts[0] == 2 and len(counts) > 1 and counts[1] == 2:  score += 15_000_000
    elif counts[0] == 2:                                          score += 5_000_000
    if max(sc.values()) >= len(cards):                            score += 10_000_000
    return score


def score_kept(cards):
    """Score a set of kept cards — exact for 5+, heuristic for fewer."""
    if len(cards) >= 5:
        return best_5_of_n(cards)
    return partial_score(cards)


def hand_strength(card_strs):
    """Normalized hand strength for 7-card hand."""
    return best_5_of_n([E7[c] for c in card_strs]) / MAX_SCORE


# ─── Synergy Score ────────────────────────────────────────────────────────────

def synergy_score(cards):
    """How well do these cards work together? (pairs, flush draws, connectors)"""
    if not cards:
        return 0.0
    ranks = [RANK_STR.index(str(c)[0]) for c in cards]
    suits = [str(c)[1] for c in cards]
    rc = Counter(ranks)
    sc = Counter(suits)

    score = 0.0
    for cnt in rc.values():
        if   cnt == 2: score += 1.0
        elif cnt == 3: score += 3.0
        elif cnt == 4: score += 6.0

    score += max(sc.values()) * 0.4

    ur = sorted(set(ranks))
    for i in range(1, len(ur)):
        gap = ur[i] - ur[i - 1]
        if   gap == 1: score += 0.6
        elif gap == 2: score += 0.2

    return score


# ─── Anti-Synergy Penalty ────────────────────────────────────────────────────

def anti_synergy_penalty(passed_cards, opp_cards):
    """Penalty for passing cards that help the opponent's known hand.
    Higher = more damaging to us (helps opponent more)."""
    if not opp_cards:
        return 0.0

    opp_ranks = [RANK_STR.index(str(c)[0]) for c in opp_cards]
    opp_suits = [str(c)[1] for c in opp_cards]
    opp_rc = Counter(opp_ranks)
    opp_sc = Counter(opp_suits)

    penalty = 0.0
    for c in passed_cards:
        r = RANK_STR.index(str(c)[0])
        s = str(c)[1]

        # Rank match: giving opponent a pair/trips/quads boost
        rm = opp_rc.get(r, 0)
        if   rm >= 3: penalty += 8.0   # giving them quads
        elif rm == 2: penalty += 4.0   # giving them trips
        elif rm == 1: penalty += 1.5   # giving them a pair

        # Suit match: flush potential
        sm = opp_sc.get(s, 0)
        if   sm >= 4: penalty += 6.0   # completing a flush
        elif sm == 3: penalty += 2.0   # 4 to a flush
        elif sm >= 2: penalty += 0.4

        # High-card value given away
        penalty += r * 0.08

    return penalty


# ─── Pass Selection (exhaustive enumeration + anti-synergy) ──────────────────

def find_pass_idx(card_strs, n, known_opp_strs=None):
    """Enumerate ALL C(len, n) pass combinations. Choose the one that
    maximizes our retained hand while minimizing opponent benefit.
    C(7,3)=35, C(7,2)=21, C(7,1)=7 — all trivially fast."""
    cards = [E7[c] for c in card_strs]
    num   = len(cards)

    opp_cards = [E7[c] for c in known_opp_strs] if known_opp_strs else []

    best_score = -float('inf')
    best_pass  = list(range(n))

    for combo in combinations(range(num), n):
        keep = [cards[i] for i in range(num) if i not in combo]

        # Score our retained hand
        my_val = score_kept(keep) / MAX_SCORE * 50.0
        syn    = synergy_score(keep) * 0.5
        combined = my_val + syn

        # Penalize helping opponent
        if opp_cards:
            passed = [cards[i] for i in combo]
            combined -= anti_synergy_penalty(passed, opp_cards)

        if combined > best_score:
            best_score = combined
            best_pass  = list(combo)

    return sorted(best_pass)


# ─── Card Tracker (all-seen approach) ────────────────────────────────────────

class CardTracker:
    """Tracks ALL cards ever seen in our hand across exchanges.

    After all 3 exchanges, we've seen 13 of the 14 cards in play (typically).
    Opponent's hand = (all 14 in play) - (our 7 cards).
    We know 6+ of their 7 cards = near-perfect equity calculation.

    The key insight: all_seen - my_current_hand = cards opponent has (that we know).
    """

    def __init__(self):
        self.all_seen    = set()    # every card string we've ever held
        self.just_passed = set()    # cards we're about to pass (pre-exchange)
        self.pending     = False    # waiting for exchange to resolve

    def reset(self, initial_hand):
        self.all_seen = set(initial_hand)
        self.just_passed = set()
        self.pending = False

    def record_pass(self, hand, pass_indices):
        """Called right before we return ActionPass."""
        self.just_passed = {hand[i] for i in pass_indices}
        self.pending = True

    def sync(self, current_hand):
        """Called when street changes after a pass — the exchange has happened.
        current_hand now includes cards received from opponent."""
        if not self.pending:
            return
        self.all_seen |= set(current_hand)
        self.pending = False
        self.just_passed = set()

    def known_opp_cards(self, my_hand):
        """Cards we've seen that aren't in our hand = opponent has them."""
        return list(self.all_seen - set(my_hand))

    def n_unknown_opp(self, my_hand):
        """How many of opponent's 7 cards are unknown to us."""
        return 7 - len(self.known_opp_cards(my_hand))

    def exact_equity(self, hand_strs, time_left=None):
        """Compute exact winning probability using tracked information.

        n_unknown=0: perfect information, instant
        n_unknown=1: enumerate ~39 cards, ~819 evals → instant
        n_unknown=2: enumerate C(~40,2)=780 combos, ~16K evals → <50ms
        """
        my_set  = set(hand_strs)
        known_opp_list = list(self.all_seen - my_set)
        n_unknown = 7 - len(known_opp_list)

        # Only compute when feasible and time-safe
        if n_unknown > 2:
            return None
        if time_left is not None and time_left < 3.0 and n_unknown > 1:
            return None

        my_eval   = [E7[c] for c in hand_strs]
        my_score  = best_5_of_n(my_eval)
        opp_known = [E7[c] for c in known_opp_list]

        # Pool of candidate cards for opponent's unknown slots
        pool = [E7[c] for c in FULL_DECK if c not in self.all_seen]

        if n_unknown == 0:
            opp_s = best_5_of_n(opp_known)
            if   my_score > opp_s: return 1.0
            elif my_score == opp_s: return 0.5
            else:                   return 0.0

        wins = 0.0
        ties = 0.0
        total = 0

        if n_unknown == 1:
            for card in pool:
                opp_s = best_5_of_n(opp_known + [card])
                if   my_score > opp_s: wins += 1
                elif my_score == opp_s: ties += 1
                total += 1

        else:  # n_unknown == 2
            for c1, c2 in combinations(pool, 2):
                opp_s = best_5_of_n(opp_known + [c1, c2])
                if   my_score > opp_s: wins += 1
                elif my_score == opp_s: ties += 1
                total += 1

        return (wins + ties * 0.5) / total if total > 0 else 0.5


# ─── Bot ──────────────────────────────────────────────────────────────────────

class Player(BaseBot):

    def __init__(self):
        self.tracker          = CardTracker()
        self._trapped_street  = None
        self._trap_active     = False
        self.last_street      = None

    def on_hand_start(self, game_info, cs):
        self.tracker.reset(cs.my_hand)
        self._trapped_street = None
        self._trap_active    = False
        self.last_street     = None

    def on_hand_end(self, game_info, cs):
        pass

    def get_move(self, game_info, cs):
        street   = cs.street
        hand     = cs.my_hand
        time_left = game_info.time_bank

        # ── Sync tracker after exchange ──────────────────────────────
        if self.tracker.pending and street != self.last_street:
            self.tracker.sync(hand)
        self.last_street = street

        # ── Pass phases (exhaustive + anti-synergy) ──────────────────
        if street in PASS_COUNTS:
            n = PASS_COUNTS[street]

            # On DoublePass/SinglePass we know some opponent cards
            known_opp = None
            if street != 'TriplePass':
                opp = self.tracker.known_opp_cards(hand)
                if opp:
                    known_opp = opp

            indices = find_pass_idx(hand, n, known_opp)
            self.tracker.record_pass(hand, indices)
            return ActionPass(indices)

        # ── Betting phases ───────────────────────────────────────────

        strength = hand_strength(hand)

        # Compute exact equity on Betting#2 and Betting#3
        equity = None
        if street in ('Betting#2', 'Betting#3'):
            equity = self.tracker.exact_equity(hand, time_left)

        # Best available estimate
        eff = equity if equity is not None else strength

        pot  = cs.pot
        cost = cs.cost_to_call

        # ─────────────────────────────────────────────────────────────
        # EQUITY-DRIVEN DECISIONS (when exact equity is available)
        # ─────────────────────────────────────────────────────────────

        if equity is not None:

            # ── Near-certain loss: fold / check ──────────────────────
            if equity < 0.10 and cost > 0:
                if cs.can_act(ActionFold):
                    return ActionFold()
                return ActionCheck()

            # ── Very weak: check, or fold if facing large bet ────────
            if equity < 0.25:
                if cs.can_act(ActionCheck):
                    return ActionCheck()
                pot_odds = cost / (pot + cost) if (pot + cost) > 0 else 1.0
                if equity < pot_odds:
                    if cs.can_act(ActionFold):
                        return ActionFold()
                return ActionCall()

            # ── Near-certain win: max value ──────────────────────────
            if equity > 0.92:
                if cs.can_act(ActionRaise):
                    lo, hi = cs.raise_bounds
                    # Scale raise with certainty
                    frac = min(0.97, 0.82 + (equity - 0.92) * 1.9)
                    return ActionRaise(min(hi, lo + int((hi - lo) * frac)))
                return ActionCall()

            # ── Strong equity: raise with trap ───────────────────────
            if equity > 0.70:
                if cs.can_act(ActionRaise):
                    lo, hi = cs.raise_bounds
                    if cost == 0:
                        # Trap: check once, then raise
                        if not self._trap_active and street != self._trapped_street:
                            self._trap_active    = True
                            self._trapped_street = street
                            if cs.can_act(ActionCheck):
                                return ActionCheck()
                        frac = min(0.88, 0.50 + (equity - 0.70) * 1.3)
                        return ActionRaise(min(hi, lo + int((hi - lo) * frac)))
                    else:
                        self._trap_active    = False
                        self._trapped_street = None
                        frac = min(0.92, 0.60 + (equity - 0.70) * 1.1)
                        return ActionRaise(min(hi, lo + int((hi - lo) * frac)))

            # ── Moderate equity: small raise or call ─────────────────
            if equity > 0.55:
                if cost == 0 and cs.can_act(ActionRaise):
                    lo, hi = cs.raise_bounds
                    return ActionRaise(lo)
                if cost > 0:
                    return ActionCall()
                if cs.can_act(ActionCheck):
                    return ActionCheck()

            # ── Marginal equity: check/call by pot odds ──────────────
            if cs.can_act(ActionCheck):
                return ActionCheck()
            if cost > 0:
                pot_odds = cost / (pot + cost) if (pot + cost) > 0 else 1.0
                if equity > pot_odds + 0.04:
                    return ActionCall()
            if cs.can_act(ActionFold):
                return ActionFold()
            return ActionCall()

        # ─────────────────────────────────────────────────────────────
        # STRENGTH-BASED DECISIONS (Betting#1 — no equity available)
        # Preserves proven bot_pass_trap logic
        # ─────────────────────────────────────────────────────────────

        is_very_strong = strength > 0.74

        # ── Trap logic ───────────────────────────────────────────────
        if cs.can_act(ActionRaise):
            lo, hi = cs.raise_bounds

            if is_very_strong:
                if cost == 0:
                    if not self._trap_active and street != self._trapped_street:
                        self._trap_active    = True
                        self._trapped_street = street
                        if cs.can_act(ActionCheck):
                            return ActionCheck()
                    return ActionRaise(min(hi, lo + int((hi - lo) * 0.85)))
                else:
                    self._trap_active    = False
                    self._trapped_street = None
                    return ActionRaise(min(hi, lo + int((hi - lo) * 0.90)))

            elif strength > 0.55:
                return ActionRaise(lo)

        # ── Check / call / fold ──────────────────────────────────────
        if cs.can_act(ActionCheck):
            return ActionCheck()

        pot_odds = cost / (pot + cost) if (pot + cost) > 0 else 1.0

        if self._trap_active and cost > 0:
            self._trap_active = False
            if is_very_strong or strength > 0.55:
                return ActionCall()

        if strength > pot_odds + 0.06 or strength > 0.55:
            return ActionCall()

        if cs.can_act(ActionFold):
            return ActionFold()
        return ActionCall()


if __name__ == '__main__':
    run_bot(Player(), parse_args())