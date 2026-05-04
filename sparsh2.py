'''
sparsh2.py — Tournament-Grade Anaconda Hold'em Bot
====================================================
Core Edge: Full card tracking across all 3 exchange rounds.
By Betting#3 we know 5-6 of opponent's 7 cards → near-exact equity.

Strategy:
  1. Track every card given to opponent; detect returns each round.
  2. Adversarial exchange: keep best cards, pass cards that least
     help opponent's accumulating known hand (eval7-scored when ≥5).
  3. Near-exact equity using known opponent cards:
     - 0 unknowns: deterministic
     - 1 unknown:  exact enumeration (~39 evals)
     - 2 unknowns: exact enumeration (~780 evals)
     - 3+ unknowns: Monte Carlo with opponent-aware pool
  4. Aggressive value betting when equity is high (we KNOW we win).
     Disciplined folding when equity is low (we KNOW we lose).
'''

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionPass
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import random
import time
from itertools import combinations
from collections import Counter

# ── Precomputed constants ─────────────────────────────────────────────────────

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
RANK_VAL = {r: i for i, r in enumerate(RANKS)}
FULL_DECK = [r + s for r in RANKS for s in SUITS]
E7 = {c: eval7.Card(c) for c in FULL_DECK}

BIG_BLIND = 20
STARTING_STACK = 5000


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _partial_score(card_strs):
    """Heuristic score for <5 card hands (used during exchange)."""
    if not card_strs:
        return 0
    ranks = [RANK_VAL[c[0]] for c in card_strs]
    suits = [c[1] for c in card_strs]
    rc = Counter(ranks)
    sc = Counter(suits)
    counts = sorted(rc.values(), reverse=True)

    score = sum(r * 250 for r in ranks)

    # Pair / set / quad bonuses
    if   counts[0] >= 4: score += 9_000_000
    elif counts[0] == 3: score += 2_800_000
    elif counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
                          score += 1_100_000
    elif counts[0] == 2:  score +=   400_000

    # Flush draw
    top_suited = sc.most_common(1)[0][1]
    if   top_suited >= 4: score += 900_000
    elif top_suited == 3: score +=  90_000
    elif top_suited == 2: score +=   9_000

    # Straight connectivity
    uniq = sorted(set(ranks))
    if len(uniq) >= 2:
        span = uniq[-1] - uniq[0]
        density = len(uniq) / (span + 1) if span > 0 else 1.0
        if   span <= 4: score += int(density * 120_000)
        elif span <= 6: score += int(density *  25_000)

    return score


def score_cards(card_strs):
    """Universal: eval7 for ≥5 cards, heuristic for <5."""
    n = len(card_strs)
    if n >= 7:
        return eval7.evaluate([E7[c] for c in card_strs])
    if n == 6:
        ev = [E7[c] for c in card_strs]
        return max(eval7.evaluate(list(cb)) for cb in combinations(ev, 5))
    if n == 5:
        return eval7.evaluate([E7[c] for c in card_strs])
    return _partial_score(card_strs)


def card_quality(card_strs):
    """Average normalized rank of cards (0..1)."""
    if not card_strs:
        return 0.5
    return sum(RANK_VAL[c[0]] for c in card_strs) / (12.0 * len(card_strs))


# ── Bot ───────────────────────────────────────────────────────────────────────

class Player(BaseBot):
    """Card-tracking Anaconda bot with near-exact equity computation."""

    # Adversarial penalty weight per exchange phase
    # Higher alpha → more weight on hurting opponent's hand
    ALPHA = {
        'TriplePass': 0.18,   # 4 opp unknowns after this phase
        'DoublePass': 0.38,   # 2 opp unknowns after this phase
        'SinglePass': 0.55,   # 1 opp unknown  after this phase
    }
    NPASS = {'TriplePass': 3, 'DoublePass': 2, 'SinglePass': 1}

    def __init__(self):
        self.hands_played = 0
        self.opp_folds = 0
        self.opp_raise_actions = 0
        self.opp_total_actions = 0
        self._reset_hand()

    def _reset_hand(self):
        self.cards_given = set()       # cards known to be in opponent's hand
        self.hand_snapshot = None      # hand before most recent exchange
        self.just_passed = set()       # cards we passed in most recent exchange
        self.pending_sync = False      # need to process exchange result
        self.last_street = None
        self.prev_hand_set = set()
        self.opp_card_quality = 0.5
        self.equity_cache = {}
        self.exchange_cache = {}
        self.we_folded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState):
        self._reset_hand()
        self.prev_hand_set = set(current_state.my_hand)

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState):
        self.hands_played += 1
        opp_hand = current_state.opp_hand
        if current_state.payoff > 0 and not self.we_folded and not opp_hand:
            self.opp_folds += 1

    # ── Card tracking ─────────────────────────────────────────────────────────

    def _sync_after_exchange(self, current_hand_set):
        """After an exchange resolves, detect which cards were returned."""
        if not self.pending_sync or self.hand_snapshot is None:
            return
        expected_kept = self.hand_snapshot - self.just_passed
        received = current_hand_set - expected_kept
        # Cards we previously gave that opponent sent back
        returned = received & self.cards_given
        self.cards_given -= returned
        self.pending_sync = False
        self.hand_snapshot = None
        self.just_passed = set()

    def _update_received_signal(self, current_hand_set):
        """Infer how strong opponent's kept range is from cards they passed us."""
        if self.prev_hand_set:
            received = list(current_hand_set - self.prev_hand_set)
            if received:
                self.opp_card_quality = card_quality(received)
        self.prev_hand_set = set(current_hand_set)

    def _record_pass(self, hand, pass_indices):
        self.hand_snapshot = set(hand)
        self.just_passed = {hand[i] for i in pass_indices}
        self.cards_given |= self.just_passed
        self.pending_sync = True

    # ── Opponent model ────────────────────────────────────────────────────────

    @property
    def opp_fold_rate(self):
        if self.hands_played < 15:
            return 0.15
        return self.opp_folds / max(1, self.hands_played)

    @property
    def opp_aggression(self):
        if self.opp_total_actions < 15:
            return 0.40
        return self.opp_raise_actions / max(1, self.opp_total_actions)

    def _infer_opp_action(self, state):
        """Track whether opponent put us to a decision with money to call."""
        cost = state.cost_to_call
        if cost <= 0:
            return

        self.opp_total_actions += 1

        # Ignore blind mismatch at start of Betting#1.
        if state.street == 'Betting#1' and state.my_wager == BIG_BLIND // 2 and state.opp_wager == BIG_BLIND:
            return

        self.opp_raise_actions += 1

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        street = current_state.street
        hand_set = set(current_state.my_hand)

        # Sync tracking when street transitions after an exchange
        if self.pending_sync and street != self.last_street:
            self._sync_after_exchange(hand_set)

        self._update_received_signal(hand_set)
        self.last_street = street

        if street in self.NPASS:
            return self._exchange(current_state, game_info)

        self._infer_opp_action(current_state)
        return self._bet(game_info, current_state)

    # ══════════════════════════════════════════════════════════════════════════
    #  EXCHANGE STRATEGY
    # ══════════════════════════════════════════════════════════════════════════

    def _exchange(self, state, game_info):
        hand = list(state.my_hand)
        n = len(hand)
        num_pass = self.NPASS[state.street]
        alpha = self.ALPHA[state.street]
        known_opp = list(self.cards_given)

        cache_key = (state.street, tuple(sorted(hand)), tuple(sorted(known_opp)))
        cached_indices = self.exchange_cache.get(cache_key)
        if cached_indices is not None:
            self._record_pass(hand, list(cached_indices))
            return ActionPass(list(cached_indices))

        tb = game_info.time_bank
        if tb > 12:
            sims = 96
        elif tb > 8:
            sims = 64
        elif tb > 4:
            sims = 36
        elif tb > 2:
            sims = 16
        else:
            sims = 6

        if state.street == 'DoublePass':
            sims = int(sims * 0.82)
        elif state.street == 'SinglePass':
            sims = int(sims * 0.62)
        sims = max(4, sims)

        my_set = set(hand)
        incoming_pool = [c for c in FULL_DECK if c not in my_set]
        incoming_samples = [random.sample(incoming_pool, num_pass) for _ in range(sims)]

        best_score = float('-inf')
        best_indices = list(range(num_pass))

        for combo in combinations(range(n), num_pass):
            kept = [hand[i] for i in range(n) if i not in combo]
            given = [hand[i] for i in combo]

            my_val = 0.0
            for incoming in incoming_samples:
                my_val += score_cards(kept + incoming)
            my_val /= max(1, sims)

            opp_accumulated = known_opp + given
            opp_val = score_cards(opp_accumulated) if opp_accumulated else 0

            pass_penalty = card_quality(given) * (80_000 + 40_000 * alpha)

            net = my_val - alpha * opp_val - pass_penalty
            if net > best_score:
                best_score = net
                best_indices = sorted(list(combo))

        self._record_pass(hand, best_indices)
        self.exchange_cache[cache_key] = tuple(best_indices)

        return ActionPass(best_indices)

    # ══════════════════════════════════════════════════════════════════════════
    #  EQUITY COMPUTATION
    # ══════════════════════════════════════════════════════════════════════════

    def _mc_equity_samples(self, my_score, known_opp_eval, pool, n_unknown, sims, deadline):
        wins = ties = trials = 0
        for _ in range(sims):
            if trials > 2 and time.perf_counter() >= deadline:
                break
            sample = random.sample(pool, n_unknown)
            opp_s = eval7.evaluate(known_opp_eval + sample)
            if my_score > opp_s:
                wins += 1
            elif my_score == opp_s:
                ties += 1
            trials += 1
        return wins, ties, trials

    def _compute_equity(self, hand_strs, game_info, state):
        if game_info.time_bank < 0.10:
            return self._fast_equity(hand_strs)

        pressure_bucket = min(
            5,
            int(6 * (state.cost_to_call / max(1, state.pot + state.cost_to_call)))
        )
        cache_key = (
            state.street,
            tuple(sorted(hand_strs)),
            tuple(sorted(self.cards_given)),
            pressure_bucket,
        )
        cached = self.equity_cache.get(cache_key)
        if cached is not None:
            return cached

        my_eval = [E7[c] for c in hand_strs]
        my_score = eval7.evaluate(my_eval)

        known_opp_strs = list(self.cards_given)
        known_opp_eval = [E7[c] for c in known_opp_strs]
        n_unknown = 7 - len(known_opp_strs)

        excluded = set(hand_strs) | self.cards_given
        pool = [E7[c] for c in FULL_DECK if c not in excluded]

        # ── Fully determined: instant ─────────────────────────────────────
        if n_unknown == 0:
            opp_s = eval7.evaluate(known_opp_eval)
            equity = 1.0 if my_score > opp_s else (0.5 if my_score == opp_s else 0.0)
            self.equity_cache[cache_key] = equity
            return equity

        # ── 1 unknown: exact enumeration (~39 evals, <0.1ms) ─────────────
        if n_unknown == 1:
            wins = ties = 0.0
            for card in pool:
                opp_s = eval7.evaluate(known_opp_eval + [card])
                if   my_score > opp_s: wins += 1
                elif my_score == opp_s: ties += 0.5
            total = len(pool)
            equity = (wins + ties) / total if total > 0 else 0.5
            self.equity_cache[cache_key] = equity
            return equity

        # ── 2 unknowns: exact enumeration (C(~40,2)≈780, <1ms) ──────────
        if n_unknown == 2:
            wins = ties = 0.0
            count = 0
            for c1, c2 in combinations(pool, 2):
                opp_s = eval7.evaluate(known_opp_eval + [c1, c2])
                if   my_score > opp_s: wins += 1
                elif my_score == opp_s: ties += 0.5
                count += 1
            equity = (wins + ties) / count if count > 0 else 0.5
            self.equity_cache[cache_key] = equity
            return equity

        # ── 3+ unknowns: Monte Carlo with deadline ──────────────────────
        tb = game_info.time_bank
        sims = 230 if tb > 12 else 160 if tb > 8 else 96 if tb > 4 else 48 if tb > 2 else 24

        if n_unknown >= 4:
            sims = int(sims * 1.12)

        pressure = state.cost_to_call / max(1, state.pot + state.cost_to_call)
        if pressure > 0.60:
            sims = int(sims * 1.20)
        elif pressure > 0.35:
            sims = int(sims * 1.08)

        remaining_rounds = max(1, 1001 - game_info.round_num)
        safe_bank = max(0.0, tb - 0.10)
        budget = max(0.0005, min(0.020, safe_bank / max(1.0, remaining_rounds * 3.0)))
        deadline = time.perf_counter() + budget

        wins, ties, trials = self._mc_equity_samples(
            my_score, known_opp_eval, pool, n_unknown, sims, deadline
        )

        if trials == 0:
            return self._fast_equity(hand_strs)

        equity = (wins + 0.5 * ties) / trials

        if tb > 4 and time.perf_counter() < deadline:
            pot_odds = state.cost_to_call / max(1, state.pot + state.cost_to_call)
            thresholds = [0.38, 0.54, 0.66, 0.74, 0.85, pot_odds, pot_odds + 0.04]
            edge = min(abs(equity - t) for t in thresholds)
            if edge < 0.03:
                extra = 150 if tb > 12 else 96 if tb > 8 else 56 if tb > 4 else 0
                if n_unknown >= 4:
                    extra = int(extra * 1.10)
                if extra > 0:
                    ew, et, etrials = self._mc_equity_samples(
                        my_score, known_opp_eval, pool, n_unknown, extra, deadline
                    )
                    if etrials > 0:
                        wins += ew
                        ties += et
                        trials += etrials
                        equity = (wins + 0.5 * ties) / trials

        # If opponent passed us high cards, their retained range tends weaker.
        info_weight = min(1.0, n_unknown / 4.0)
        equity += (self.opp_card_quality - 0.5) * 0.08 * info_weight
        equity = max(0.01, min(0.99, equity))

        self.equity_cache[cache_key] = equity
        return equity

    def _fast_equity(self, hand_strs):
        """Rank-based heuristic when no time for computation."""
        q = sum(RANK_VAL[c[0]] for c in hand_strs) / 84.0
        return max(0.10, min(0.90, 0.18 + 0.62 * q))

    # ══════════════════════════════════════════════════════════════════════════
    #  BETTING STRATEGY
    # ══════════════════════════════════════════════════════════════════════════

    def _bet(self, game_info, state):
        equity = self._compute_equity(list(state.my_hand), game_info, state)

        if state.street == 'Betting#1':
            equity = equity * 0.90 + 0.05
        elif state.street == 'Betting#2':
            equity = equity * 0.96 + 0.02

        pot = state.pot
        cost = state.cost_to_call

        if cost > 0:
            return self._facing_bet(state, equity, pot, cost)
        return self._leading(state, equity, pot)

    def _make_raise(self, state, pot, frac_lo, frac_hi):
        """Raise by frac of pot on top of opponent's wager. Returns None if can't raise."""
        if not state.can_act(ActionRaise):
            return None
        mn, mx = state.raise_bounds
        additional = int(pot * random.uniform(frac_lo, frac_hi))
        target = max(mn, min(mx, state.opp_wager + additional))
        return ActionRaise(target)

    # ── Facing a bet / raise ──────────────────────────────────────────────────

    def _facing_bet(self, s, equity, pot, cost):
        pot_odds = cost / (pot + cost) if (pot + cost) > 0 else 0.0
        ofr = self.opp_fold_rate
        agg = self.opp_aggression
        call_edge = max(0.012, min(0.075, 0.048 - (agg - 0.40) * 0.06))

        # Near-certain winner: raise big
        if equity >= 0.92:
            r = self._make_raise(s, pot, 0.80, 1.15)
            if r:
                return r
            return ActionCall()

        # Strong hand: raise sometimes, always call
        if equity >= 0.77:
            raise_prob = 0.36 + 0.20 * max(0.0, ofr - 0.20) - 0.10 * max(0.0, agg - 0.45)
            raise_prob = max(0.18, min(0.58, raise_prob))
            if random.random() < raise_prob:
                r = self._make_raise(s, pot, 0.50, 0.80)
                if r:
                    return r
            return ActionCall() if s.can_act(ActionCall) else ActionCheck()

        # Decent hand: call if +EV
        if equity > pot_odds + call_edge:
            return ActionCall() if s.can_act(ActionCall) else ActionCheck()

        # Marginal: call small bets
        max_frac = 0.32 + 0.14 * (1.0 - agg)
        if equity > pot_odds - 0.01 and cost <= pot * max_frac:
            return ActionCall() if s.can_act(ActionCall) else ActionCheck()

        # Occasional bluff re-raise vs timid opponent
        bluff_reraise = 0.03 + 0.10 * max(0.0, ofr - 0.30)
        bluff_reraise *= 0.85 if agg > 0.55 else 1.0
        if ofr > 0.27 and equity < 0.30 and random.random() < min(0.09, bluff_reraise):
            r = self._make_raise(s, pot, 0.65, 0.90)
            if r:
                return r

        # Fold
        if s.can_act(ActionFold):
            self.we_folded = True
            return ActionFold()
        return ActionCall() if s.can_act(ActionCall) else ActionCheck()

    # ── Acting first (check or bet) ───────────────────────────────────────────

    def _leading(self, s, equity, pot):
        ofr = self.opp_fold_rate
        agg = self.opp_aggression

        # Monster: bet big
        if equity >= 0.87:
            if random.random() < 0.14 and s.can_act(ActionCheck):
                return ActionCheck()
            r = self._make_raise(s, pot, 0.70, 1.05)
            if r:
                return r

        # Strong: value bet
        if equity >= 0.71:
            r = self._make_raise(s, pot, 0.45, 0.70)
            if r:
                return r

        # Above average: bet sometimes
        if equity >= 0.57:
            open_prob = 0.34 + 0.12 * max(0.0, ofr - 0.20) - 0.08 * max(0.0, agg - 0.50)
            open_prob = max(0.20, min(0.55, open_prob))
            if random.random() < open_prob:
                r = self._make_raise(s, pot, 0.28, 0.48)
                if r:
                    return r
            return ActionCheck() if s.can_act(ActionCheck) else ActionCall()

        # Bluff with weak hands
        if equity < 0.36:
            bluff_prob = 0.04 + 0.35 * max(0.0, ofr - 0.25)
            if agg > 0.55:
                bluff_prob *= 0.72
            elif agg < 0.32:
                bluff_prob *= 1.18
            bluff_prob = max(0.02, min(0.20, bluff_prob))
            if random.random() < bluff_prob:
                r = self._make_raise(s, pot, 0.55, 0.85)
                if r:
                    return r

        # Default: check
        if s.can_act(ActionCheck):
            return ActionCheck()
        if s.can_act(ActionCall):
            return ActionCall()
        if s.can_act(ActionFold):
            self.we_folded = True
            return ActionFold()
        return ActionCall()


if __name__ == '__main__':
    run_bot(Player(), parse_args())