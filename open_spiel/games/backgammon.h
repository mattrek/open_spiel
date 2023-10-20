// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_BACKGAMMON_H_
#define OPEN_SPIEL_GAMES_BACKGAMMON_H_

#include <array>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// An implementation of the classic: https://en.wikipedia.org/wiki/Backgammon
// using rule set from
// http://usbgf.org/learn-backgammon/backgammon-rules-and-terms/rules-of-backgammon/
// where red -> 'x' (player 0) and white -> 'o' (player 1).
//
// Currently does not support the doubling cube nor "matches" (multiple games
// where outcomes are scored and tallied to 21).
//
// Parameters:
//   "hyper_backgammon"  bool    Use Hyper-backgammon variant [1] (def: false)
//   "scoring_type"      string  Type of scoring for the game: "winloss_scoring"
//                               (default), "enable_gammons", or "full_scoring"
//
// [1] https://bkgm.com/variants/HyperBackgammon.html. Hyper-backgammon is a
// simplified backgammon start setup which is small enough to solve. Note that
// it is not the full Hyper-backgammon sinc do not have cube is not implemented.

namespace open_spiel {
namespace backgammon {

constexpr bool kUseResnet = false;

inline constexpr const int kNumPlayers = 2;
inline constexpr const int kNumChanceOutcomes = 21;
inline constexpr const int kNumPoints = 24;
inline constexpr const int kNumDice = 2;
inline constexpr const int kNumDiceOutcomes = 6;
inline constexpr const int kXPlayerId = 0;
inline constexpr const int kOPlayerId = 1;
inline constexpr const int kPassPos = -1;

// Number of checkers per player in the standard game. For varaints, use
// BackgammonGame::NumCheckersPerPlayer.
inline constexpr const int kNumCheckersPerPlayer = 15;

// TODO: look into whether these can be set to 25 and -2 to avoid having a
// separate helper function (PositionToStringHumanReadable) to convert moves
// to strings.
inline constexpr const int kBarPos = 100;
inline constexpr const int kScorePos = 101;

// An n-length checker sequence is encoded into an Action as an n-digit number
// in base kNumSingleCheckerActions+1.
inline constexpr const int kNumMovesPerCheckerSequence = 4;

// The set of actions consists of both checker moves and some other actions.
// For checker moves, each checker on a point or the bar can be moved according
// to a single die, yielding (kNumPoints + 1) * kNumDiceOutcomes unique actions.
inline constexpr const int kNumSingleCheckerActions =
    (kNumPoints + 1) * kNumDiceOutcomes;
constexpr const int ipow(int a, int b) { return b == 0 ? 1 : a * ipow(a, b-1); }
inline constexpr const int kNumCheckerActions =
    ipow(kNumSingleCheckerActions+1, kNumMovesPerCheckerSequence);

// The action encoding stores a number in { 1, ..., kNumDistinctActions }.
// The first kNumCheckerActions of these encode checker moves, and the
// remaining encode each of the other actions as enumerated below.
inline constexpr const Action kEndTurnAction = kNumCheckerActions + 1;
inline constexpr const Action kRollAction = kNumCheckerActions + 2;
inline constexpr const Action kDoubleAction = kNumCheckerActions + 3;
inline constexpr const Action kTakeAction = kNumCheckerActions + 4;
inline constexpr const Action kDropAction = kNumCheckerActions + 5;
inline constexpr const int kNumDistinctActions = kNumCheckerActions + 5;

// See ObservationTensorShape for details.
inline constexpr const int kBoardEncodingSize = kNumPoints * kNumPlayers;
// TODO: cleanup state encoding size for resent, value below is for mlp
inline constexpr const int kStateEncodingSize = 420;
inline constexpr const char* kDefaultScoringType = "winloss_scoring";
inline constexpr bool kDefaultHyperBackgammon = false;

// Game scoring type, whether to score gammons/backgammons specially.
enum class ScoringType {
  kWinLossScoring,  // "winloss_scoring": Score only 1 point per player win.
  kEnableGammons,   // "enable_gammons": Score 2 points for a "gammon".
  kFullScoring,     // "full_scoring": Score gammons as well as 3 points for a
                    // "backgammon".
};

// The number of dice (i.e. up to 4 for doublets) that must be legally played
// in a state.
// Rule 2 in Movement of Checkers:
// A player must use both numbers of a roll if this is legally possible (or
// all four numbers of a double). When only one number can be played, the
// player must play that number. Or if either number can be played but not
// both, the player must play the larger one. When neither number can be used,
// the player loses his turn. In the case of doubles, when all four numbers
// cannot be played, the player must play as many numbers as he can.
enum class LegalLevel {
  kNoDice,
  kLowDie,
  kHighDie,
  kTwoDice,
  kThreeDice,
  kFourDice,
};

struct CheckerMove {
  // Pass is encoded as (pos, num, hit) = (-1, -1, false).
  int pos;  // 0-24  (0-23 for locations on the board and kBarPos)
  int num;  // 1-6
  bool hit;
  CheckerMove(int _pos, int _num, bool _hit)
      : pos(_pos), num(_num), hit(_hit) {}
  bool operator<(const CheckerMove& rhs) const {
    return (pos * 6 + (num - 1)) < (rhs.pos * 6 + rhs.num - 1);
  }
};

// This is a small helper to track historical turn info not stored in the moves.
// It is only needed for proper implementation of Undo.
struct TurnHistoryInfo {
  int player;
  int prev_player;
  std::vector<int> dice;
  std::vector<int> remaining_dice;
  Action action;
  bool double_turn;
  bool first_move_hit;
  bool second_move_hit;
  TurnHistoryInfo(int _player, int _prev_player, std::vector<int> _dice,
                  std::vector<int> _remaining_dice,
                  int _action, bool _double_turn, bool fmh, bool smh)
      : player(_player),
        prev_player(_prev_player),
        dice(_dice),
        remaining_dice(_remaining_dice),
        action(_action),
        double_turn(_double_turn),
        first_move_hit(fmh),
        second_move_hit(smh) {}
};

class BackgammonGame;
class CheckerMoveSequence;

class BackgammonState : public State {
 public:
  BackgammonState(const BackgammonState&) = default;
  BackgammonState(std::shared_ptr<const Game>, ScoringType scoring_type,
                  bool hyper_backgammone);

  Player CurrentPlayer() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ActionToMatString(Action action) const;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  std::string PositionId() const;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;

  // Setter function used for debugging and tests. Note: this does not set the
  // historical information properly, so Undo likely will not work on states
  // set this way!
  void SetState(int cur_player, bool double_turn, const std::vector<int>& dice,
                const std::vector<int>& bar, const std::vector<int>& scores,
                const std::vector<std::vector<int>>& board);

  // Returns the opponent of the specified player.
  int Opponent(int player) const;

  // Compute a distance between 'from' and 'to'. The from can be kBarPos. The
  // to can be a number below 0 or above 23, but do not use kScorePos directly.
  /* mattrek: unused code
  int GetDistance(int player, int from, int to) const;
  */

  // Is this position off the board, i.e. >23 or <0?
  bool IsOff(int player, int pos) const;

  // Returns whether pos2 is further (closer to scoring) than pos1 for the
  // specifed player.
  /* mattrek: unused code
  bool IsFurther(int player, int pos1, int pos2) const;
  */

  // Is this a legal from -> to checker move? Here, the to_pos can be a number
  // that is outside {0, ..., 23}; if so, it is counted as "off the board" for
  // the corresponding player (i.e. >23 is a bear-off move for XPlayerId, and
  // <0 is a bear-off move for OPlayerId).
  /* mattrek: unused code
  bool IsLegalFromTo(int player, int from_pos, int to_pos, int my_checkers_from,
                     int opp_checkers_to) const;
  */

  // Get the To position for this play given the from position and number of
  // pips on the die. This function simply adds the values: the return value
  // will be a position that might be off the the board (<0 or >23).
  int GetToPos(int player, int from_pos, int pips) const;


  // Count the total number of checkers for this player (on the board, in the
  // bar, and have borne off). Should be 15 for the standard game.
  int CountTotalCheckers(int player) const;

  // Returns if moving from the position for the number of spaces is a hit.
  bool IsHit(Player player, int from_pos, int num) const;

  // Accessor functions for some of the specific data.
  /* mattrek: unused code
  int player_turns() const { return turns_; }
  int player_turns(int player) const {
    return (player == kXPlayerId ? x_turns_ : o_turns_);
  }
  */
  int bar(int player) const { return bar_[player]; }
  int score(int player) const { return scores_[player]; }
  int dice(int i) const { return dice_[i]; }
  int remaining_dice(int i) const { return remaining_dice_[i]; }
  bool double_turn() const { return double_turn_; }

  // Get the number of checkers on the board in the specified position belonging
  // to the specified player. The position can be kBarPos or any valid position
  // on the main part of the board, but kScorePos (use score() to get the number
  // of checkers born off).
  int board(int player, int pos) const;

  // Action encoding / decoding functions. Note, the converted checker moves
  // do not contain the hit information; use the AddHitInfo function to get the
  // hit information.
  Action SingleCheckerMoveToSpielMove(const CheckerMove& move) const;
  Action CheckerMovesToSpielMove(const std::vector<CheckerMove>& moves) const;
  CheckerMove SpielMoveToSingleCheckerMove(Action action) const;
  std::vector<CheckerMove> SpielMoveToCheckerMoves(Action action) const;
  bool ApplyCheckerMove(const CheckerMove& move);
  void UndoCheckerMove(const CheckerMove& move);

  // Return checker moves with extra hit information.
  std::vector<CheckerMove>
  AugmentWithHitInfo(Player player,
                     const std::vector<CheckerMove> &cmoves) const;
  // Declared public for testing purposes.
  LegalLevel DetermineLegalLevel() const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  void SetupInitialBoard();
  void RollDice(int outcome);
  void InitRemainingDice();

  /* mattrek: unused code
  bool IsPosInHome(int player, int pos) const;
  */
  bool AllInHome(int player) const;
  /* mattrek: unused code
  int CheckersInHome(int player) const;
  */
  bool UsableDiceOutcome(int outcome) const;
  int PositionFromBar(int player, int spaces) const;
  int PositionFrom(int player, int pos, int spaces) const;
  int NumOppCheckers(int player, int pos) const;
  std::string DiceToString(int outcome) const;
  int IsGammoned(int player) const;
  int IsBackgammoned(int player) const;
  int DiceValue(int i) const;

  Action EncodedBarMove() const;

  // A helper function used by ActionToString to add necessary hit information
  // and compute whether the move goes off the board.
  int AugmentCheckerMove(CheckerMove* cmove, int player, int start) const;

  // Returns the position of the furthest checker in the home of this player.
  // Returns -1 if none found.
  int FurthestCheckerInHome(int player) const;

  std::set<CheckerMoveSequence> LegalCheckerMoveSequences() const;
  std::set<CheckerMove> LegalSingleCheckerMoves() const;
  std::set<CheckerMove> SingleCheckerMoves(
      int die, bool bFirstOnly = false) const;
  // Helper function for DetermineLegalLevel(), returns the max number of dice that
  // can be played from 'state' given 'dice_to_play'.
  int NumMaxPlayableDies(BackgammonState* state, std::vector<int> dice_to_play) const;

  ScoringType scoring_type_;  // Which rules apply when scoring the game.
  bool hyper_backgammon_;     // Is the Hyper-backgammon variant enabled?

  Player cur_player_;
  Player prev_player_;
  int turns_;
  int x_turns_;
  int o_turns_;
  bool double_turn_;
  std::vector<int> dice_;    // The 2 rolled dice.
  std::vector<int> remaining_dice_;    // Dice (up to 4) remaining to play, adjusted for LegalLevel..
  std::vector<int> bar_;     // Checkers of each player in the bar.
  std::vector<int> scores_;  // Checkers returned home by each player.
  std::vector<std::vector<int>> board_;  // Checkers for each player on points.
  std::vector<TurnHistoryInfo> turn_history_info_;  // Info needed for Undo.
};

class BackgammonGame : public Game {
 public:
  explicit BackgammonGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new BackgammonState(
        shared_from_this(), scoring_type_, hyper_backgammon_));
  }

  // On the first turn there are 30 outcomes: 15 for each player (rolls without
  // the doubles).
  int MaxChanceOutcomes() const override { return 30; }

  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return 1000; }

  // Upper bound: chance node per move, with an initial chance node for
  // determining starting player.
  int MaxChanceNodesInHistory() const override { return MaxGameLength() + 1; }

  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return -MaxUtility(); }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override;

  std::vector<int> ObservationTensorShape() const override {
    if (kUseResnet) {
      // plane 1 for X checkers (25->0, i.e.: bar + board + off)
      // plane 2 for O checkers (0->25, i.e.: off + board + bar)
      // plane 3 for X to act
      // plane 4 for O to act
      // plane 5 for num remaining 1s to play
      // plane 6 for num remaining 2s to play
      // plane 7 for num remaining 3s to play
      // plane 8 for num remaining 4s to play
      // plane 9 for num remaining 5s to play
      // plane 10 for num remaining 6s to play
      // plane 11 for X away score
      // plane 12 for O away score
      // plane 13 for crawford score
      // plane 14 for cube level
      // plane 15 for dice have rolled
      // plane 16 for cube was turned
      return {16, 1, kNumPoints + 2};
    }

    // 2x191 for 2 players:
    // - 1x7 one-hot w overage for num checkers on bar
    // - 24x7 one-hot w overage for num checkers on a point
    // - 1x16 one-hot for checkers off
    // X turn (0 or 1).
    // O turn (0 or 1).
    // 6x5 for num remaining of each die (1s thru 6s) as a one-hot
    // X away score == 1
    // O away score == 1
    // crawford score == 0
    // cube level == 1
    // dice have rolled (0 or 1)
    // cube was turned == 0
    return {kStateEncodingSize};
  }

  int NumCheckersPerPlayer() const;

 private:
  ScoringType scoring_type_;  // Which rules apply when scoring the game.
  bool hyper_backgammon_;     // Is hyper-backgammon variant enabled?
};

class CheckerMoveSequence {
public:
  CheckerMoveSequence(const BackgammonState& state);
  BackgammonState GetState() const { return state_; }
  std::vector<CheckerMove> GetMoves() const { return moves_; }
  std::string GetId() const { return id_; };
  void AddMove(const CheckerMove& move);
  bool operator<(const CheckerMoveSequence& rhs) const {
    return GetId() < rhs.GetId();
  }

private:
  std::vector<CheckerMove> moves_; // sequence of moves
  BackgammonState state_; // resulting state after applying moves
  std::string id_; // a unique id of the resulting position
};

}  // namespace backgammon
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BACKGAMMON_H_
