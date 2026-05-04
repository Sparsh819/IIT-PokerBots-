"""
EMPEROR — Ultimate Anaconda Hold'em Bot
=========================================
Built on KING v3 (bot_pass_trap architecture) with 3 surgical edges:

 1. Exhaustive pass search — enumerate all C(7,n) subsets vs greedy removal
 2. Poison-aware passing  — penalize giving opponent coordinated/high cards
 3. Bidirectional card tracking — track cards received from opponent

Design philosophy: keep pass improvements, keep exact-equity overrides,
and use KING's proven betting core for stable tournament EV.
All computation is O(1) heuristic or small exhaustive search.
Total per-decision budget: < 1ms.
"""

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionPass
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
from itertools import combinations
from collections import Counter

# ─── Constants ────────────────────────────────────────────────────────────────

RANK_STR   = '23456789TJQKA'
RANK_VAL   = {r: i for i, r in enumerate(RANK_STR)}
SUITS      = 'cdhs'
FULL_DECK  = [r + s for r in RANK_STR for s in SUITS]
E7         = {c: eval7.Card(c) for c in FULL_DECK}

MAX_SCORE    = 135_004_167
PASS_COUNTS  = {'TriplePass': 3, 'DoublePass': 2, 'SinglePass': 1}

# ─── Core Hand Evaluation ────────────────────────────────────────────────────

def parse_cards(card_strs):
    return [eval7.Card(c) for c in card_strs]

def best_5_of_n(cards):
    if len(cards) == 5:
        return eval7.evaluate(cards)
    return max(eval7.evaluate(list(c)) for c in combinations(cards, 5))

def hand_strength(card_strs):
    return best_5_of_n(parse_cards(card_strs)) / MAX_SCORE


# ─── Partial Hand Scoring (for <5 cards during pass eval) ─────────────────────

def score_kept(cards):
    n = len(cards)
    if n >= 5:
        return best_5_of_n(cards)
    if n == 0:
        return 0

    ranks  = sorted([RANK_STR.index(str(c)[0]) for c in cards], reverse=True)
    suits  = [str(c)[1] for c in cards]
    rc, sc = Counter(ranks), Counter(suits)
    counts = sorted(rc.values(), reverse=True)

    score = sum(r * 5_000 for r in ranks)

    # Pair/trip/quad bonuses
    if   counts[0] == 4:                                        score += 80_000_000
    elif counts[0] == 3:                                        score += 25_000_000
    elif counts[0] == 2 and len(counts) > 1 and counts[1] == 2: score += 15_000_000
    elif counts[0] == 2:                                        score += 5_000_000

    # Flush draw potential
    if n >= 3:
        max_suit = max(sc.values())
        if max_suit == n:       score += 12_000_000   # mono-suited
        elif max_suit == n - 1: score += 6_000_000    # one off flush

    # Straight connectivity
    ur = sorted(set(ranks))
    consec = 1
    max_consec = 1
    for i in range(1, len(ur)):
        if ur[i] - ur[i-1] == 1:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 1
    if max_consec >= 3: score += 4_000_000
    elif max_consec >= 2: score += 1_500_000

    return score


# ─── Synergy Score ────────────────────────────────────────────────────────────

def synergy_score(cards, card_strs):
    if not cards:
        return 0.0
    ranks = [RANK_STR.index(str(c)[0]) for c in cards]
    suits = [str(c)[1] for c in cards]
    rc = Counter(ranks)
    sc = Counter(suits)

    score = 0.0

    # Pair/trip/quad synergy
    for cnt in rc.values():
        if cnt == 2:   score += 1.2
        elif cnt == 3: score += 3.5
        elif cnt == 4: score += 7.0

    # Flush draw synergy
    max_suit = max(sc.values())
    if max_suit >= 4:   score += 4.0
    elif max_suit == 3: score += 1.8
    else:               score += max_suit * 0.3

    # Straight draw synergy
    ur = sorted(set(ranks))
    for i in range(1, len(ur)):
        gap = ur[i] - ur[i - 1]
        if gap == 1:   score += 0.7
        elif gap == 2: score += 0.25

    # High card bonus
    for r in ranks:
        if r >= 12: score += 0.3   # Ace
        elif r >= 11: score += 0.15  # King

    return score


# ─── Poison Score — penalize passing cards that help opponent ─────────────────

def poison_score(passed_strs, cards_already_given):
    """
    How much do the cards we're about to pass help the opponent?
    Higher = worse (don't pass these).
    """
    if not passed_strs:
        return 0.0

    score = 0.0
    passed_ranks = [RANK_VAL[c[0]] for c in passed_strs]
    passed_suits = [c[1] for c in passed_strs]

    # Penalty for passing high cards
    for r in passed_ranks:
        if r >= 12: score += 2.0    # Ace
        elif r >= 11: score += 1.2  # King
        elif r >= 10: score += 0.6  # Queen

    # Penalty for passing cards that match RANKS of previously given cards
    # (gives opponent pairs/trips/quads)
    if cards_already_given:
        given_ranks = Counter(RANK_VAL[c[0]] for c in cards_already_given)
        for r in passed_ranks:
            if r in given_ranks:
                score += 3.0 * given_ranks[r]  # each matching card is very bad

        # Penalty for passing cards in same SUIT as previously given
        # (helps opponent build flushes)
        given_suits = Counter(c[1] for c in cards_already_given)
        for s in passed_suits:
            if s in given_suits:
                score += 1.5 * given_suits[s]

    # Penalty for passing internally coordinated cards
    # (e.g. passing two hearts together gives opponent a flush draw)
    pass_suit_cnt = Counter(passed_suits)
    for cnt in pass_suit_cnt.values():
        if cnt >= 2: score += cnt * 0.8

    pass_rank_cnt = Counter(passed_ranks)
    for cnt in pass_rank_cnt.values():
        if cnt >= 2: score += cnt * 1.5  # giving opponent a pair outright

    return score


# ─── Exhaustive Pass Selection ───────────────────────────────────────────────

def find_pass_idx(card_strs, n, cards_already_given=None):
    """
    Enumerate all C(len,n) subsets to pass. Pick the one that maximizes
    (value of kept cards) - (poison of passed cards).
    Much better than greedy removal for n=3 (35 combos) and n=2 (21 combos).
    """
    cards = parse_cards(card_strs)
    num_cards = len(cards)

    if cards_already_given is None:
        cards_already_given = set()

    best_score  = -float('inf')
    best_pass   = list(range(n))  # fallback: pass first n

    for pass_combo in combinations(range(num_cards), n):
        pass_set    = set(pass_combo)
        kept_cards  = [cards[i] for i in range(num_cards) if i not in pass_set]
        kept_strs   = [card_strs[i] for i in range(num_cards) if i not in pass_set]
        passed_strs = [card_strs[i] for i in pass_combo]

        # Value of what we keep
        keep_eq  = score_kept(kept_cards) / MAX_SCORE if len(kept_cards) >= 4 else 0
        keep_syn = synergy_score(kept_cards, kept_strs)

        # Cost of what we give
        poison = poison_score(passed_strs, cards_already_given)

        # Combined: heavily weight keep value, moderate synergy, light poison
        combined = keep_eq * 50.0 + keep_syn * 0.6 - poison * 0.35

        if combined > best_score:
            best_score = combined
            best_pass  = sorted(pass_combo)

    return best_pass


# ─── Card Tracker (enhanced: bidirectional) ──────────────────────────────────

class CardTracker:
    """Track cards in both directions for maximum information."""

    def __init__(self):
        self.cards_given   = set()   # cards we passed to opponent (that they still have)
        self.cards_received_total = set()  # all cards we ever received from opponent
        self.hand_snapshot = None
        self.just_passed   = set()
        self.pending_sync  = False

    def reset(self):
        self.cards_given.clear()
        self.cards_received_total.clear()
        self.hand_snapshot = None
        self.just_passed.clear()
        self.pending_sync = False

    def sync_after_exchange(self, current_hand_set):
        """After an exchange, figure out what we received and update tracking."""
        if not self.pending_sync or self.hand_snapshot is None:
            return

        expected_kept = self.hand_snapshot - self.just_passed
        received = current_hand_set - expected_kept
        self.cards_received_total |= received

        # If opponent sent us back cards we gave them, remove from cards_given
        returned = received & self.cards_given
        self.cards_given -= returned

        self.pending_sync  = False
        self.hand_snapshot = None
        self.just_passed.clear()

    def record_pass(self, hand, pass_indices):
        self.hand_snapshot = set(hand)
        self.just_passed   = {hand[i] for i in pass_indices}
        self.cards_given  |= self.just_passed
        self.pending_sync  = True

    def exact_equity(self, hand_strs):
        """Compute exact equity when we know enough of opponent's cards."""
        known_opp   = list(self.cards_given)
        n_unknown   = 7 - len(known_opp)

        if n_unknown > 2:
            return None

        my_eval   = [E7[c] for c in hand_strs]
        my_score  = eval7.evaluate(my_eval)
        known_e7  = [E7[c] for c in known_opp]

        excluded = set(hand_strs) | self.cards_given
        pool     = [E7[c] for c in FULL_DECK if c not in excluded]

        if n_unknown == 0:
            opp_s = eval7.evaluate(known_e7)
            return 1.0 if my_score > opp_s else (0.5 if my_score == opp_s else 0.0)

        if n_unknown == 1:
            wins = ties = 0.0
            for card in pool:
                opp_s = eval7.evaluate(known_e7 + [card])
                if my_score > opp_s:   wins += 1
                elif my_score == opp_s: ties += 0.5
            total = len(pool)
            return (wins + ties) / total if total > 0 else 0.5

        # n_unknown == 2
        wins = ties = 0.0
        count = 0
        for c1, c2 in combinations(pool, 2):
            opp_s = eval7.evaluate(known_e7 + [c1, c2])
            if my_score > opp_s:   wins += 1
            elif my_score == opp_s: ties += 0.5
            count += 1
        return (wins + ties) / count if count > 0 else 0.5


# ─── Bot ──────────────────────────────────────────────────────────────────────

class Player(BaseBot):

    def __init__(self):
        self.tracker  = CardTracker()

        # Trap state (from king.py architecture)
        self._trapped_street = None
        self._trap_active    = False
        self.last_street     = None

    # ── Round lifecycle ──────────────────────────────────────────────────

    def on_hand_start(self, game_info, cs):
        self.tracker.reset()
        self._trapped_street = None
        self._trap_active    = False
        self.last_street     = None

    def on_hand_end(self, game_info, cs):
        pass

    # ── Main decision function ───────────────────────────────────────────

    def get_move(self, game_info, cs):
        street   = cs.street
        hand_set = set(cs.my_hand)

        # Sync card tracking after exchange
        if self.tracker.pending_sync and street != self.last_street:
            self.tracker.sync_after_exchange(hand_set)
        self.last_street = street

        # ── Pass phases ──────────────────────────────────────────────────
        if street in PASS_COUNTS:
            n = PASS_COUNTS[street]
            indices = find_pass_idx(
                cs.my_hand, n,
                cards_already_given=self.tracker.cards_given
            )
            self.tracker.record_pass(cs.my_hand, indices)
            return ActionPass(indices)

        # ── Betting phases ───────────────────────────────────────────────

        strength = hand_strength(cs.my_hand)

        # Exact equity from card tracking (Betting#2 and #3)
        tracked = None
        if street in ('Betting#2', 'Betting#3'):
            tracked = self.tracker.exact_equity(cs.my_hand)

        pot  = cs.pot
        cost = cs.cost_to_call
        is_very_strong = strength > 0.74

        # ── BINARY OVERRIDE: fold when certain to lose ───────────────────
        if tracked is not None and tracked < 0.08 and cost > 0:
            if cs.can_act(ActionFold):
                return ActionFold()
            if cs.can_act(ActionCheck):
                return ActionCheck()

        # ── BINARY OVERRIDE: max value when certain to win ───────────────
        if tracked is not None and tracked > 0.92:
            if cs.can_act(ActionRaise):
                lo, hi = cs.raise_bounds
                return ActionRaise(min(hi, lo + int((hi - lo) * 0.90)))
            return ActionCall()

        # ── Trap logic (KING core) ───────────────────────────────────────
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

        # ── Check / call / fold ──────────────────────────────────────────
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