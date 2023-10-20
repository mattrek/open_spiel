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

#include "open_spiel/games/backgammon.h"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace backgammon {
namespace {

// A few constants to help with the conversion to human-readable string formats.
// TODO: remove these once we've changed kBarPos and kScorePos (see TODO in
// header).
constexpr int kNumBarPosHumanReadable = 25;
constexpr int kNumOffPosHumanReadable = -2;

const std::vector<std::pair<Action, double>> kChanceOutcomes = {
    std::pair<Action, double>(1, 1.0 / 18),
    std::pair<Action, double>(2, 1.0 / 18),
    std::pair<Action, double>(3, 1.0 / 18),
    std::pair<Action, double>(4, 1.0 / 18),
    std::pair<Action, double>(5, 1.0 / 18),
    std::pair<Action, double>(6, 1.0 / 18),
    std::pair<Action, double>(7, 1.0 / 18),
    std::pair<Action, double>(8, 1.0 / 18),
    std::pair<Action, double>(9, 1.0 / 18),
    std::pair<Action, double>(10, 1.0 / 18),
    std::pair<Action, double>(11, 1.0 / 18),
    std::pair<Action, double>(12, 1.0 / 18),
    std::pair<Action, double>(13, 1.0 / 18),
    std::pair<Action, double>(14, 1.0 / 18),
    std::pair<Action, double>(15, 1.0 / 18),
    std::pair<Action, double>(16, 1.0 / 36),
    std::pair<Action, double>(17, 1.0 / 36),
    std::pair<Action, double>(18, 1.0 / 36),
    std::pair<Action, double>(19, 1.0 / 36),
    std::pair<Action, double>(20, 1.0 / 36),
    std::pair<Action, double>(21, 1.0 / 36),
};

// Doubles not allowed for the initial roll to determine who goes first.
// Range 1-15: X goes first, range 16-30: O goes first.
const std::vector<std::pair<Action, double>> kFirstRollChanceOutcomes = {
    std::pair<Action, double>(1, 1.0 / 30),
    std::pair<Action, double>(2, 1.0 / 30),
    std::pair<Action, double>(3, 1.0 / 30),
    std::pair<Action, double>(4, 1.0 / 30),
    std::pair<Action, double>(5, 1.0 / 30),
    std::pair<Action, double>(6, 1.0 / 30),
    std::pair<Action, double>(7, 1.0 / 30),
    std::pair<Action, double>(8, 1.0 / 30),
    std::pair<Action, double>(9, 1.0 / 30),
    std::pair<Action, double>(10, 1.0 / 30),
    std::pair<Action, double>(11, 1.0 / 30),
    std::pair<Action, double>(12, 1.0 / 30),
    std::pair<Action, double>(13, 1.0 / 30),
    std::pair<Action, double>(14, 1.0 / 30),
    std::pair<Action, double>(15, 1.0 / 30),
    std::pair<Action, double>(16, 1.0 / 30),
    std::pair<Action, double>(17, 1.0 / 30),
    std::pair<Action, double>(18, 1.0 / 30),
    std::pair<Action, double>(19, 1.0 / 30),
    std::pair<Action, double>(20, 1.0 / 30),
    std::pair<Action, double>(21, 1.0 / 30),
    std::pair<Action, double>(22, 1.0 / 30),
    std::pair<Action, double>(23, 1.0 / 30),
    std::pair<Action, double>(24, 1.0 / 30),
    std::pair<Action, double>(25, 1.0 / 30),
    std::pair<Action, double>(26, 1.0 / 30),
    std::pair<Action, double>(27, 1.0 / 30),
    std::pair<Action, double>(28, 1.0 / 30),
    std::pair<Action, double>(29, 1.0 / 30),
    std::pair<Action, double>(30, 1.0 / 30),
};

const std::vector<std::vector<int>> kChanceOutcomeValues = {
    {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 3}, {2, 4},
    {2, 5}, {2, 6}, {3, 4}, {3, 5}, {3, 6}, {4, 5}, {4, 6},
    {5, 6}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};

int NumCheckersPerPlayer(const Game* game) {
  return static_cast<const BackgammonGame*>(game)->NumCheckersPerPlayer();
}

// Facts about the game
const GameType kGameType{
    /*short_name=*/"backgammon",
    /*long_name=*/"Backgammon",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*min_num_players=*/2,
    /*max_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"hyper_backgammon", GameParameter(kDefaultHyperBackgammon)},
     {"scoring_type",
      GameParameter(static_cast<std::string>(kDefaultScoringType))}}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BackgammonGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

ScoringType ParseScoringType(const std::string& st_str) {
  if (st_str == "winloss_scoring") {
    return ScoringType::kWinLossScoring;
  } else if (st_str == "enable_gammons") {
    return ScoringType::kEnableGammons;
  } else if (st_str == "full_scoring") {
    return ScoringType::kFullScoring;
  } else {
    SpielFatalError("Unrecognized scoring_type parameter: " + st_str);
  }
}

std::string PositionToString(int pos) {
  switch (pos) {
    case kBarPos:
      return "Bar";
    case kScorePos:
      return "Score";
    case -1:
      return "Pass";
    default:
      return absl::StrCat(pos);
  }
}

std::string CurPlayerToString(Player cur_player) {
  switch (cur_player) {
    case kXPlayerId:
      return "X";
    case kOPlayerId:
      return "O";
    case kChancePlayerId:
      return "*";
    case kTerminalPlayerId:
      return "T";
    default:
      SpielFatalError(absl::StrCat("Unrecognized player id: ", cur_player));
  }
}

std::string PositionToStringHumanReadable(int pos) {
  if (pos == kNumBarPosHumanReadable) {
    return "Bar";
  } else if (pos == kNumOffPosHumanReadable) {
    return "Off";
  } else {
    return PositionToString(pos);
  }
}

int BackgammonState::AugmentCheckerMove(CheckerMove* cmove, int player,
                                        int start) const {
  int end = cmove->num;
  if (end != kPassPos) {
    // Not a pass, so work out where the piece finished
    end = start - cmove->num;
    if (end <= 0) {
      end = kNumOffPosHumanReadable;  // Off
    } else if (board_[Opponent(player)]
                     [player == kOPlayerId ? (end - 1) : (kNumPoints - end)] ==
               1) {
      cmove->hit = true;  // Check to see if move is a hit
    }
  }
  return end;
}

std::string BackgammonState::ActionToString(Player player,
                                            Action action) const {
  if (player == kChancePlayerId) {
    if (turns_ >= 0) {
      // Normal chance roll.
      return absl::StrCat("chance outcome ", action,
                          " (roll: ", kChanceOutcomeValues[action-1][1],
                          kChanceOutcomeValues[action-1][0], ")");
    } else {
      // Initial roll to determine who starts.
      const char* starter = (action <= 15 ? "X starts" : "O starts");
      if (action > 15) {
        action -= 15;
      }
      return absl::StrCat("chance outcome ", action, " ", starter, ", ",
                          "(roll: ", kChanceOutcomeValues[action-1][1],
                          kChanceOutcomeValues[action-1][0], ")");
    }
  } else if (action > kNumCheckerActions) {
    switch (action) {
      case kEndTurnAction: return "EndTurn";
      case kRollAction: return "Roll";
      case kDoubleAction: return "Double";
      case kTakeAction: return "Take";
      case kDropAction: return "Drop";
      default:
        SpielFatalError(
            absl::StrCat("Unexpected action in ActionToString(): ", action));
    }
  } else {
    // Assemble a human-readable string representation of the move using
    // standard backgammon notation:
    //
    // - Always show the numbering going from Bar->24->0->Off, irrespective of
    //   which player is moving.
    // - Show the start position followed by end position.
    // - Show hits with an asterisk, e.g. 9/7*.
    // - Order the moves by highest number first, e.g. 22/7 10/8 not 10/8 22/7.
    //   Not an official requirement, but seems to be standard convention.
    // - Show duplicate moves as 10/8(2).
    // - Show moves on a single piece as 10/8/5 not 10/8 8/5
    //
    // Note that there are tests to ensure the ActionToString follows this
    // output format. Any changes would need to be reflected in the tests as
    // well.

    /*
    std::vector<CheckerMove> cmoves = SpielMoveToCheckerMoves(player, action);

    int cmove0_start;
    int cmove1_start;
    if (player == kOPlayerId) {
      cmove0_start = (cmoves[0].pos == kBarPos ? kNumBarPosHumanReadable
                                               : cmoves[0].pos + 1);
      cmove1_start = (cmoves[1].pos == kBarPos ? kNumBarPosHumanReadable
                                               : cmoves[1].pos + 1);
    } else {
      // swap the board numbering round for Player X so player is moving
      // from 24->0
      cmove0_start = (cmoves[0].pos == kBarPos ? kNumBarPosHumanReadable
                                               : kNumPoints - cmoves[0].pos);
      cmove1_start = (cmoves[1].pos == kBarPos ? kNumBarPosHumanReadable
                                               : kNumPoints - cmoves[1].pos);
    }

    // Add hit information and compute whether the moves go off the board.
    int cmove0_end = AugmentCheckerMove(&cmoves[0], player, cmove0_start);
    int cmove1_end = AugmentCheckerMove(&cmoves[1], player, cmove1_start);

    // check for 2 pieces hitting on the same point.
    bool double_hit =
        (cmoves[1].hit && cmoves[0].hit && cmove1_end == cmove0_end);

    std::string returnVal = "";
    if (cmove0_start == cmove1_start &&
        cmove0_end == cmove1_end) {     // same move, show as (2).
      if (cmoves[1].num == kPassPos) {  // Player can't move at all!
        returnVal = "Pass";
      } else {
        returnVal = absl::StrCat(action, " - ",
                                 PositionToStringHumanReadable(cmove0_start),
                                 "/", PositionToStringHumanReadable(cmove0_end),
                                 cmoves[0].hit ? "*" : "", "(2)");
      }
    } else if ((cmove0_start < cmove1_start ||
                (cmove0_start == cmove1_start && cmove0_end < cmove1_end) ||
                cmoves[0].num == kPassPos) &&
               cmoves[1].num != kPassPos) {
      // tradition to start with higher numbers first,
      // so swap moves round if this not the case. If
      // there is a pass move, put it last.
      if (cmove1_end == cmove0_start) {
        // Check to see if the same piece is moving for both
        // moves, as this changes the format of the output.
        returnVal = absl::StrCat(
            action, " - ", PositionToStringHumanReadable(cmove1_start), "/",
            PositionToStringHumanReadable(cmove1_end), cmoves[1].hit ? "*" : "",
            "/", PositionToStringHumanReadable(cmove0_end),
            cmoves[0].hit ? "*" : "");
      } else {
        returnVal = absl::StrCat(
            action, " - ", PositionToStringHumanReadable(cmove1_start), "/",
            PositionToStringHumanReadable(cmove1_end), cmoves[1].hit ? "*" : "",
            " ",
            (cmoves[0].num != kPassPos)
                ? PositionToStringHumanReadable(cmove0_start)
                : "",
            (cmoves[0].num != kPassPos) ? "/" : "",
            PositionToStringHumanReadable(cmove0_end),
            (cmoves[0].hit && !double_hit) ? "*" : "");
      }
    } else {
      if (cmove0_end == cmove1_start) {
        // Check to see if the same piece is moving for both
        // moves, as this changes the format of the output.
        returnVal = absl::StrCat(
            action, " - ", PositionToStringHumanReadable(cmove0_start), "/",
            PositionToStringHumanReadable(cmove0_end), cmoves[0].hit ? "*" : "",
            "/", PositionToStringHumanReadable(cmove1_end),
            cmoves[1].hit ? "*" : "");
      } else {
        returnVal = absl::StrCat(
            action, " - ", PositionToStringHumanReadable(cmove0_start), "/",
            PositionToStringHumanReadable(cmove0_end), cmoves[0].hit ? "*" : "",
            " ",
            (cmoves[1].num != kPassPos)
                ? PositionToStringHumanReadable(cmove1_start)
                : "",
            (cmoves[1].num != kPassPos) ? "/" : "",
            PositionToStringHumanReadable(cmove1_end),
            (cmoves[1].hit && !double_hit) ? "*" : "");
      }
    }
    */

    std::string moves_str;
    std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(action);
    std::unique_ptr<State> cstate = this->Clone();
    BackgammonState* state = dynamic_cast<BackgammonState*>(cstate.get());
    for(CheckerMove move : moves) {
      int move_start;
      if (player == kOPlayerId) {
        move_start = (move.pos == kBarPos ? kNumBarPosHumanReadable
                                           : move.pos + 1);
      } else {
        // swap the board numbering round for Player X so player is moving
        // from 24->0
        move_start = (move.pos == kBarPos ? kNumBarPosHumanReadable
                                          : kNumPoints - move.pos);
      }

      // Add hit information and compute whether the moves go off the board.
      int move_end = state->AugmentCheckerMove(&move, player, move_start);
      state->ApplyCheckerMove(move);
      moves_str = absl::StrCat(moves_str, " ",
          PositionToStringHumanReadable(move_start), "/",
          PositionToStringHumanReadable(move_end), move.hit ? "*" : "");
    }

    return absl::StrCat(action, " -", moves_str);
  }
}

std::string BackgammonState::ActionToMatString(Action action) const {
  Player player = CurrentPlayer();
  if (player == kChancePlayerId) {
    if (turns_ < 0 && action > 15) {
      action -= 15;
    }
    // return the dice roll
    return absl::StrFormat("%d%d:",
        kChanceOutcomeValues[action-1][1],
        kChanceOutcomeValues[action-1][0]);
  }
  if (action > kNumCheckerActions) {
    switch (action) {
      case kEndTurnAction: return "";
      case kRollAction: return "";
      //case kDoubleAction: return "Double";
      //case kTakeAction: return "Take";
      //case kDropAction: return "Drop";
      default:
        SpielFatalError(
            absl::StrCat("Unexpected action in ActionToMatString(): ", action));
    }
  }

  std::string moves_str;
  std::unique_ptr<State> cstate = this->Clone();
  BackgammonState* state = dynamic_cast<BackgammonState*>(cstate.get());
  for(CheckerMove move : SpielMoveToCheckerMoves(action)) {
    int move_start;
    if (player == kOPlayerId) {
      move_start = (move.pos == kBarPos ? kNumBarPosHumanReadable
                                         : move.pos + 1);
    } else {
      // swap the board numbering round for Player X so player is moving
      // from 24->0
      move_start = (move.pos == kBarPos ? kNumBarPosHumanReadable
                                        : kNumPoints - move.pos);
    }

    // Add hit information and compute whether the moves go off the board.
    int move_end = state->AugmentCheckerMove(&move, player, move_start);
    state->ApplyCheckerMove(move);
    moves_str = absl::StrCat(moves_str, " ",
        PositionToStringHumanReadable(move_start), "/",
        PositionToStringHumanReadable(move_end), move.hit ? "*" : "");
  }
  return moves_str;
}

std::string BackgammonState::InformationStateString(Player player) const {
  // mattrek: Only implemented so that treeviz_example.py will work.
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string BackgammonState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void onehot(absl::Span<float>::iterator& it, int size, int val) {
  for (int i = 0; i <= size; i++) {
    *it++ = (i == val) ? 1.0 : 0.0;
  }
}

void onehot_plus_overage(
    absl::Span<float>::iterator& it, int size, int maxval, int val) {
  for (int i = 0; i <= size; i++) {
    *it++ = (i == val) ? 1.0 : 0.0;
  }
  *it++ = (val <= size) ? 0.0 : 1.0 + (val - size) / (1.0 * (maxval - size));
}

void BackgammonState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_TRUE(player == CurrentPlayer());

  if (kUseResnet) {
    int num_planes = 16;
    int num_inputs_per_plane = kNumPoints + 2;
    SPIEL_CHECK_EQ(values.size(), num_planes * num_inputs_per_plane);
    TensorView<2> view(values, {num_planes, num_inputs_per_plane}, true);
    int plane_idx = 0;

    if (open_spiel::kPlayerCentricObs) {
      bool invert = player != 0;

      // plane 1 for X checkers (25->0, i.e.: bar + board + off)
      view[{plane_idx, 0}] = bar_[player] / 15.0;
      for (int i = 0; i < kNumPoints; i++) {
        view[{plane_idx, i + 1}] = board_[player][invert ? 23 - i : i] / 15.0;
      }
      view[{plane_idx, kNumPoints + 1}] = scores_[player] / 15.0;
      plane_idx++;

      // plane 2 for O checkers (0->25, i.e.: off + board + bar)
      view[{plane_idx, 0}] = scores_[1 - player] / 15.0;
      for (int i = 0; i < kNumPoints; i++) {
        view[{plane_idx, i + 1}] = board_[1 - player][invert ? 23 - i : i] / 15.0;
      }
      view[{plane_idx, kNumPoints + 1}] = bar_[1 - player] / 15.0;
      plane_idx++;

      // plane 3 for X to act
      for (int i = 0; i < num_inputs_per_plane; i++) {
        view[{plane_idx, i}] = 1;
      }
      plane_idx++;

      // plane 4 for O to act
      for (int i = 0; i < num_inputs_per_plane; i++) {
        view[{plane_idx, i}] = 0;
      }
      plane_idx++;

      // plane 5 (thru 10) for num remaining 1s (thru 6s) to play
      for (int j = 0; j < kNumDiceOutcomes; j++) {
        for (int i = 0; i < num_inputs_per_plane; i++) {
          view[{plane_idx, i}] = dice_.empty() ? 0 : remaining_dice_[j];
        }
        plane_idx++;
      }

      // plane 11 for X away score
      for (int i = 0; i < num_inputs_per_plane; i++) {
        view[{plane_idx, i}] = 1;
      }
      plane_idx++;

      // plane 12 for O away score
      for (int i = 0; i < num_inputs_per_plane; i++) {
        view[{plane_idx, i}] = 1;
      }
      plane_idx++;

      // plane 13 for crawford score
      for (int i = 0; i < num_inputs_per_plane; i++) {
        view[{plane_idx, i}] = 0;
      }
      plane_idx++;

      // plane 14 for cube level
      for (int i = 0; i < num_inputs_per_plane; i++) {
        view[{plane_idx, i}] = 1;
      }
      plane_idx++;

      // plane 15 for dice have rolled
      for (int i = 0; i < num_inputs_per_plane; i++) {
        view[{plane_idx, i}] = dice_.empty() ? 0 : 1;
      }
      plane_idx++;

      // plane 16 for cube was turned
      for (int i = 0; i < num_inputs_per_plane; i++) {
        view[{plane_idx, i}] = 0;
      }
      plane_idx++;
      SPIEL_CHECK_EQ(plane_idx, num_planes);
      return;
    }  // end if (kPlayerCentricObs)

    // plane 1 for X checkers (25->0, i.e.: bar + board + off)
    view[{plane_idx, 0}] = bar_[0] / 15.0;
    for (int i = 0; i < kNumPoints; i++) {
      view[{plane_idx, i + 1}] = board_[0][i] / 15.0;
    }
    view[{plane_idx, kNumPoints + 1}] = scores_[0] / 15.0;
    plane_idx++;

    // plane 2 for O checkers (0->25, i.e.: off + board + bar)
    view[{plane_idx, 0}] = scores_[1] / 15.0;
    for (int i = 0; i < kNumPoints; i++) {
      view[{plane_idx, i + 1}] = board_[1][i] / 15.0;
    }
    view[{plane_idx, kNumPoints + 1}] = bar_[1] / 15.0;
    plane_idx++;

    // plane 3 for X to act
    for (int i = 0; i < num_inputs_per_plane; i++) {
      view[{plane_idx, i}] = (cur_player_ == 0) ? 1 : 0;
    }
    plane_idx++;

    // plane 4 for O to act
    for (int i = 0; i < num_inputs_per_plane; i++) {
      view[{plane_idx, i}] = (cur_player_ == 1) ? 1 : 0;
    }
    plane_idx++;

    // plane 5 (thru 10) for num remaining 1s (thru 6s) to play
    for (int j = 0; j < kNumDiceOutcomes; j++) {
      for (int i = 0; i < num_inputs_per_plane; i++) {
        view[{plane_idx, i}] = dice_.empty() ? 0 : remaining_dice_[j];
      }
      plane_idx++;
    }

    // plane 11 for X away score
    for (int i = 0; i < num_inputs_per_plane; i++) {
      view[{plane_idx, i}] = 1;
    }
    plane_idx++;

    // plane 12 for O away score
    for (int i = 0; i < num_inputs_per_plane; i++) {
      view[{plane_idx, i}] = 1;
    }
    plane_idx++;

    // plane 13 for crawford score
    for (int i = 0; i < num_inputs_per_plane; i++) {
      view[{plane_idx, i}] = 0;
    }
    plane_idx++;

    // plane 14 for cube level
    for (int i = 0; i < num_inputs_per_plane; i++) {
      view[{plane_idx, i}] = 1;
    }
    plane_idx++;

    // plane 15 for dice have rolled
    for (int i = 0; i < num_inputs_per_plane; i++) {
      view[{plane_idx, i}] = dice_.empty() ? 0 : 1;
    }
    plane_idx++;

    // plane 16 for cube was turned
    for (int i = 0; i < num_inputs_per_plane; i++) {
      view[{plane_idx, i}] = 0;
    }
    plane_idx++;
    SPIEL_CHECK_EQ(plane_idx, num_planes);
    return;
  }  // end if (kUseResnet)

  SPIEL_CHECK_EQ(values.size(), kStateEncodingSize);
  auto value_it = values.begin();
  // The format of this vector is described in Section 3.4 of "G. Tesauro,
  // Practical issues in temporal-difference learning, 1994."
  // https://link.springer.com/article/10.1007/BF00992697
  /*
  for (int count : board_[player]) {
    *value_it++ = ((count == 1) ? 1 : 0);
    *value_it++ = ((count == 2) ? 1 : 0);
    *value_it++ = ((count == 3) ? 1 : 0);
    *value_it++ = ((count > 3) ? (count - 3) : 0);
  }
  for (int count : board_[opponent]) {
    *value_it++ = ((count == 1) ? 1 : 0);
    *value_it++ = ((count == 2) ? 1 : 0);
    *value_it++ = ((count == 3) ? 1 : 0);
    *value_it++ = ((count > 3) ? (count - 3) : 0);
  }
  *value_it++ = (bar_[player]);
  *value_it++ = (scores_[player]);
  *value_it++ = ((cur_player_ == player) ? 1 : 0);

  *value_it++ = (bar_[opponent]);
  *value_it++ = (scores_[opponent]);
  *value_it++ = ((cur_player_ == opponent) ? 1 : 0);
  */

  if (open_spiel::kPlayerCentricObs) {
    bool invert = player != 0;
    for (int plyr = 0; plyr < kNumPlayers; plyr++) {
      if (invert) {
        onehot_plus_overage(value_it, 5, 15, bar_[1 - plyr]);
        for (int i = 0; i < 24; i++) {
          onehot_plus_overage(value_it, 5, 15, board_[1 - plyr][23 - i]);
        }
        onehot(value_it, 15, scores_[1 - plyr]);

      } else {
        onehot_plus_overage(value_it, 5, 15, bar_[plyr]);
        for (int count : board_[plyr]) {
          onehot_plus_overage(value_it, 5, 15, count);
        }
        onehot(value_it, 15, scores_[plyr]);
      }
    }
    *value_it++ = 1; //((cur_player_ == 0) ? 1 : 0);
    *value_it++ = 0; //((cur_player_ == 1) ? 1 : 0);

    // num remaining 1s (thru 6s) to play
    for (int j = 0; j < kNumDiceOutcomes; j++) {
      onehot(value_it, 4, dice_.empty() ? 0 : remaining_dice_[j]);
    }

    // X away score
    *value_it++ = 1;

    // O away score
    *value_it++ = 1;

    // crawford game?
    *value_it++ = 0;

    // cube level
    *value_it++ = 1;

    // dice have rolled?
    *value_it++ = dice_.empty() ? 0 : 1;

    // cube was offered?
    *value_it++ = 0;

    SPIEL_CHECK_EQ(value_it, values.end());
    return;
  }  // end if {kPlayerCentricObs}

  for (int plyr = 0; plyr < kNumPlayers; plyr++) {
    onehot_plus_overage(value_it, 5, 15, bar_[plyr]);
    for (int count : board_[plyr]) {
      onehot_plus_overage(value_it, 5, 15, count);
    }
    onehot(value_it, 15, scores_[plyr]);
  }
  *value_it++ = ((cur_player_ == 0) ? 1 : 0);
  *value_it++ = ((cur_player_ == 1) ? 1 : 0);

  // num remaining 1s (thru 6s) to play
  for (int j = 0; j < kNumDiceOutcomes; j++) {
    onehot(value_it, 4, dice_.empty() ? 0 : remaining_dice_[j]);
  }

  // X away score
  *value_it++ = 1;

  // O away score
  *value_it++ = 1;

  // crawford game?
  *value_it++ = 0;

  // cube level
  *value_it++ = 1;

  // dice have rolled?
  *value_it++ = dice_.empty() ? 0 : 1;

  // cube was offered?
  *value_it++ = 0;

  SPIEL_CHECK_EQ(value_it, values.end());
}

BackgammonState::BackgammonState(std::shared_ptr<const Game> game,
                                 ScoringType scoring_type,
                                 bool hyper_backgammon)
    : State(game),
      scoring_type_(scoring_type),
      hyper_backgammon_(hyper_backgammon),
      cur_player_(kChancePlayerId),
      prev_player_(kChancePlayerId),
      turns_(-1),
      x_turns_(0),
      o_turns_(0),
      double_turn_(false),
      dice_({}),
      remaining_dice_({}),
      bar_({0, 0}),
      scores_({0, 0}),
      board_(
          {std::vector<int>(kNumPoints, 0), std::vector<int>(kNumPoints, 0)}),
      turn_history_info_({}) {
  SetupInitialBoard();
}

void BackgammonState::SetupInitialBoard() {
  if (hyper_backgammon_) {
    // https://bkgm.com/variants/HyperBackgammon.html
    // Each player has one checker on each of the furthest points.
    board_[kXPlayerId][0] = board_[kXPlayerId][1] = board_[kXPlayerId][2] = 1;
    board_[kOPlayerId][23] = board_[kOPlayerId][22] = board_[kOPlayerId][21] =
        1;
  } else {
   /*
    board_[kXPlayerId][0] = 2;
    board_[kXPlayerId][1] = 2;
    board_[kXPlayerId][2] = 2;
    board_[kXPlayerId][6] = 1;
    board_[kXPlayerId][8] = 1;
    board_[kXPlayerId][9] = 2;
    board_[kXPlayerId][12] = 2;
    board_[kXPlayerId][13] = 2;
    board_[kXPlayerId][23] = 1;
    // OPlayer.
    board_[kOPlayerId][4] = 4;
    board_[kOPlayerId][5] = 2;
    turns_ = 1;
    cur_player_ = kXPlayerId;
    prev_player_ = kOPlayerId;
    dice_ = {2, 2};
    remaining_dice_ = {0, 4, 0, 0, 0, 0};
    scores_ = {0, 9};
    return;
    */

    // mattrek: input conditioning: make sure max inputs are seen
    std::mt19937 rng(absl::ToUnixNanos(absl::Now()));
    int idx = 500 * absl::Uniform(rng, 0.0, 1.0);
    if (idx < 24) {
      board_[kXPlayerId][idx] = board_[kOPlayerId][23-idx] = 15;
    } else if (idx < 25) {
      bar_[0] = bar_[1] = 15;
    } else {
      // Setup the board. First, XPlayer.
      board_[kXPlayerId][0] = 2;
      board_[kXPlayerId][11] = 5;
      board_[kXPlayerId][16] = 3;
      board_[kXPlayerId][18] = 5;
      // OPlayer.
      board_[kOPlayerId][23] = 2;
      board_[kOPlayerId][12] = 5;
      board_[kOPlayerId][7] = 3;
      board_[kOPlayerId][5] = 5;
      /*
      board_[kXPlayerId][18] = 5;
      board_[kXPlayerId][19] = 5;
      board_[kXPlayerId][20] = 5;
      board_[kOPlayerId][5] = 5;
      board_[kOPlayerId][4] = 5;
      board_[kOPlayerId][3] = 5;
      */
    }
  }
}

int BackgammonState::board(int player, int pos) const {
  if (pos == kBarPos) {
    return bar_[player];
  } else {
    SPIEL_CHECK_GE(pos, 0);
    SPIEL_CHECK_LT(pos, kNumPoints);
    return board_[player][pos];
  }
}

Player BackgammonState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
}

int BackgammonState::Opponent(int player) const { return 1 - player; }

void BackgammonState::RollDice(int action) {
  dice_.push_back(kChanceOutcomeValues[action-1][0]);
  dice_.push_back(kChanceOutcomeValues[action-1][1]);
  InitRemainingDice();
}

int BackgammonState::DiceValue(int i) const {
  SPIEL_CHECK_GE(i, 0);
  SPIEL_CHECK_LT(i, dice_.size());

  if (dice_[i] >= 1 && dice_[i] <= 6) {
    return dice_[i];
  } else {
    SpielFatalError(absl::StrCat("Bad dice value: ", dice_[i]));
  }
}

void BackgammonState::InitRemainingDice() {
  if (dice_.empty()) return;

  int hi_die = std::max(dice_[0], dice_[1]);
  int lo_die = std::min(dice_[0], dice_[1]);
  remaining_dice_.assign(kNumDiceOutcomes, 0);
  switch (DetermineLegalLevel()) {
    case LegalLevel::kLowDie:
      remaining_dice_[lo_die - 1]++;
      break;
    case LegalLevel::kHighDie:
      remaining_dice_[hi_die - 1]++;
      break;
    case LegalLevel::kTwoDice:
      remaining_dice_[lo_die - 1]++;
      remaining_dice_[hi_die - 1]++;
      break;
    case LegalLevel::kThreeDice:
      remaining_dice_[hi_die - 1] = 3;
      break;
    case LegalLevel::kFourDice:
      remaining_dice_[hi_die - 1] = 4;
      break;
    default:
      break;
  }
}

void BackgammonState::DoApplyAction(Action action) {
  SPIEL_CHECK_GT(action, 0);
  SPIEL_CHECK_LE(action, kNumDistinctActions);
  if (IsChanceNode()) {
    turn_history_info_.push_back(
        TurnHistoryInfo(kChancePlayerId, prev_player_,
                        dice_, remaining_dice_, action, double_turn_,
                        false, false));

    if (turns_ == -1) {
      SPIEL_CHECK_TRUE(dice_.empty());
      if (action <= 15) {
        // X starts.
        cur_player_ = prev_player_ = kXPlayerId;
      } else {
        // O Starts
        cur_player_ = prev_player_ = kOPlayerId;
        action -= 15;
      }
      RollDice(action);
      turns_ = 0;
      return;
    } else {
      // Normal chance node.
      SPIEL_CHECK_TRUE(dice_.empty());
      // mattrek: player who chose to roll is now cur player
      //cur_player_ = Opponent(prev_player_);
      cur_player_ = prev_player_;
      RollDice(action);
      return;
    }
  }

  if (dice_.empty()) {
    // mattrek: cur_player just chose to roll
    SPIEL_CHECK_EQ(action, kRollAction);
    prev_player_ = cur_player_;
    cur_player_ = kChancePlayerId;
    return;
  }

  if (action == kEndTurnAction) {
    turns_++;
    if (cur_player_ == kXPlayerId) {
      x_turns_++;
    } else if (cur_player_ == kOPlayerId) {
      o_turns_++;
    }
    cur_player_ = Opponent(cur_player_);
    dice_.clear();
    return;
  }

  bool move_hit = false;
  for (const CheckerMove& move : SpielMoveToCheckerMoves(action)) {
    bool hit = ApplyCheckerMove(move);
    move_hit = move_hit || hit;
  }

  turn_history_info_.push_back(
      TurnHistoryInfo(cur_player_, prev_player_, dice_, remaining_dice_,
                      action, double_turn_, move_hit, false));

  prev_player_ = cur_player_;
}

void BackgammonState::UndoAction(int player, Action action) {
    //mattrek: prefer not to rely on this
    SpielFatalError("UndoAction() called!");

    /*
    const TurnHistoryInfo& thi = turn_history_info_.back();
    SPIEL_CHECK_EQ(thi.player, player);
    SPIEL_CHECK_EQ(action, thi.action);
    cur_player_ = thi.player;
    prev_player_ = thi.prev_player;
    dice_ = thi.dice;
    remaining_dice_ = thi.remaining_dice;
    double_turn_ = thi.double_turn;
    if (player != kChancePlayerId) {
      std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(player, action);
      SPIEL_CHECK_EQ(moves.size(), 2);
      moves[0].hit = thi.first_move_hit;
      moves[1].hit = thi.second_move_hit;
      UndoCheckerMove(moves[1]);
      UndoCheckerMove(moves[0]);
      if (!double_turn_) {
        turns_--;
        if (player == kXPlayerId) {
          x_turns_--;
        } else if (player == kOPlayerId) {
          o_turns_--;
        }
      }
    }
  }
  turn_history_info_.pop_back();
  history_.pop_back();
  --move_number_;
  */
}

bool BackgammonState::IsHit(Player player, int from_pos, int num) const {
  if (from_pos != kPassPos) {
    int to = PositionFrom(player, from_pos, num);
    return to != kScorePos && board(Opponent(player), to) == 1;
  } else {
    return false;
  }
}

Action BackgammonState::EncodedBarMove() const { return 24; }

Action BackgammonState::SingleCheckerMoveToSpielMove(const CheckerMove& move) const {
  SPIEL_CHECK_GE(move.pos, 0);  // kPassPos shouldnt make it to here
  SPIEL_CHECK_GE(move.num, 1);
  SPIEL_CHECK_LE(move.num, 6);

  int pos = (move.pos == kBarPos) ? EncodedBarMove()
      : (open_spiel::kPlayerCentricObs && CurrentPlayer() != 0)
        ? 23 - move.pos : move.pos;
  int action = (pos * kNumDiceOutcomes) + move.num;
  SPIEL_CHECK_GT(action, 0);
  SPIEL_CHECK_LE(action, kNumSingleCheckerActions);
  return action;
}

Action BackgammonState::CheckerMovesToSpielMove(
    const std::vector<CheckerMove>& moves) const {
  Action action = 0;
  for (const CheckerMove& move : moves) {
    action = action * (kNumSingleCheckerActions + 1)
        + SingleCheckerMoveToSpielMove(move);
  }
  return action;
}

// The given action is expected to be a checker move here.
// (i.e. other actions are not expected)
CheckerMove BackgammonState::SpielMoveToSingleCheckerMove(Action action) const {
  SPIEL_CHECK_GT(action, 0);
  SPIEL_CHECK_LE(action, kNumSingleCheckerActions);
  action--;
  int pos = action / kNumDiceOutcomes;
  if (pos == EncodedBarMove()) {
    pos = kBarPos;
  }  else if (open_spiel::kPlayerCentricObs && CurrentPlayer() != 0) {
    pos = 23 - pos;  // invert
  }

  int num = (action % kNumDiceOutcomes) + 1;
  return CheckerMove(pos, num, /*hit=*/ false);
}

std::vector<CheckerMove> BackgammonState::SpielMoveToCheckerMoves(Action action) const {
  SPIEL_CHECK_GT(action, 0);
  SPIEL_CHECK_LE(action, kNumCheckerActions);
  std::vector<CheckerMove> moves;
  while (action > 0) {
    int temp_action = action % (kNumSingleCheckerActions + 1);
    moves.insert(moves.begin(), SpielMoveToSingleCheckerMove(temp_action));
    action /= (kNumSingleCheckerActions + 1);
  }
  SPIEL_CHECK_LE(moves.size(), kNumMovesPerCheckerSequence);
  return moves;
}

bool BackgammonState::AllInHome(int player) const {
  if (bar_[player] > 0) {
    return false;
  }

  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LE(player, 1);

  // Looking for any checkers outside home.
  // --> XPlayer scans 0-17.
  // --> OPlayer scans 6-23.
  int scan_start = (player == kXPlayerId ? 0 : 6);
  int scan_end = (player == kXPlayerId ? 17 : 23);

  for (int i = scan_start; i <= scan_end; ++i) {
    if (board_[player][i] > 0) {
      return false;
    }
  }

  return true;
}

int BackgammonState::FurthestCheckerInHome(int player) const {
  // Looking for any checkers in home.
  // --> XPlayer scans 18 -> 23
  // --> OPlayer scans  5 -> 0
  int scan_start = (player == kXPlayerId ? 18 : 5);
  int scan_end = (player == kXPlayerId ? 24 : -1);
  int inc = (player == kXPlayerId ? 1 : -1);

  for (int i = scan_start; i != scan_end; i += inc) {
    if (board_[player][i] > 0) {
      return i;
    }
  }
  return -1;
}

int BackgammonState::PositionFromBar(int player, int spaces) const {
  if (player == kXPlayerId) {
    return -1 + spaces;
  } else if (player == kOPlayerId) {
    return 24 - spaces;
  } else {
    SpielFatalError(absl::StrCat("Invalid player: ", player));
  }
}

int BackgammonState::PositionFrom(int player, int pos, int spaces) const {
  if (pos == kBarPos) {
    return PositionFromBar(player, spaces);
  }

  if (player == kXPlayerId) {
    int new_pos = pos + spaces;
    return (new_pos > 23 ? kScorePos : new_pos);
  } else if (player == kOPlayerId) {
    int new_pos = pos - spaces;
    return (new_pos < 0 ? kScorePos : new_pos);
  } else {
    SpielFatalError(absl::StrCat("Invalid player: ", player));
  }
}

int BackgammonState::NumOppCheckers(int player, int pos) const {
  return board_[Opponent(player)][pos];
}

bool BackgammonState::IsOff(int player, int pos) const {
  // Returns if an absolute position is off the board.
  return ((player == kXPlayerId && pos > 23) ||
          (player == kOPlayerId && pos < 0));
}

int BackgammonState::GetToPos(int player, int from_pos, int pips) const {
  if (player == kXPlayerId) {
    return (from_pos == kBarPos ? -1 : from_pos) + pips;
  } else if (player == kOPlayerId) {
    return (from_pos == kBarPos ? 24 : from_pos) - pips;
  } else {
    SpielFatalError(absl::StrCat("Player (", player, ") unrecognized."));
  }
}

std::string BackgammonState::DiceToString(int outcome) const {
  if (outcome > 6) {
    return std::to_string(outcome - 6) + "u";
  } else {
    return std::to_string(outcome);
  }
}

int BackgammonState::CountTotalCheckers(int player) const {
  int total = 0;
  for (int i = 0; i < 24; ++i) {
    SPIEL_CHECK_GE(board_[player][i], 0);
    total += board_[player][i];
  }
  SPIEL_CHECK_GE(bar_[player], 0);
  total += bar_[player];
  SPIEL_CHECK_GE(scores_[player], 0);
  total += scores_[player];
  return total;
}

int BackgammonState::IsGammoned(int player) const {
  if (hyper_backgammon_) {
    // TODO(author5): remove this when the doubling cube is implemented.
    // In Hyper-backgammon, gammons and backgammons only multiply when the cube
    // has been offered and accepted. However, we do not yet support the cube.
    return false;
  }

  // Does the player not have any checkers borne off?
  return scores_[player] == 0;
}

int BackgammonState::IsBackgammoned(int player) const {
  if (hyper_backgammon_) {
    // TODO(author5): remove this when the doubling cube is implemented.
    // In Hyper-backgammon, gammons and backgammons only multiply when the cube
    // has been offered and accepted. However, we do not yet support the cube.
    return false;
  }

  // Does the player not have any checkers borne off and either has a checker
  // still in the bar or still in the opponent's home?
  if (scores_[player] > 0) {
    return false;
  }

  if (bar_[player] > 0) {
    return true;
  }

  // XPlayer scans 0-5.
  // OPlayer scans 18-23.
  int scan_start = (player == kXPlayerId ? 0 : 18);
  int scan_end = (player == kXPlayerId ? 5 : 23);

  for (int i = scan_start; i <= scan_end; ++i) {
    if (board_[player][i] > 0) {
      return true;
    }
  }

  return false;
}

std::set<CheckerMoveSequence> BackgammonState::LegalCheckerMoveSequences() const {
  std::set<CheckerMoveSequence> seqs;
  for (const CheckerMove& move : LegalSingleCheckerMoves()) {
    CheckerMoveSequence seq(*this);
    seq.AddMove(move);
    seqs.insert(seq);
  }
  if (seqs.empty()) { return seqs; }
  for (int i = 1; i < kNumMovesPerCheckerSequence; i++) {
    std::set<CheckerMoveSequence> new_seqs;
    for (CheckerMoveSequence seq : seqs) {
      for (const CheckerMove& move : seq.GetState().LegalSingleCheckerMoves()) {
        CheckerMoveSequence new_seq(seq);
        new_seq.AddMove(move);
        // TODO:  weed out duplicates
        new_seqs.insert(new_seq);
      }
    }
    if (new_seqs.empty()) { return seqs; }
    seqs = new_seqs;
  }
  return seqs;
}

std::set<CheckerMove> BackgammonState::LegalSingleCheckerMoves() const {
  // Since remaining_dice_ was constructed with knowledge of the required
  // LegalLevel, it will be legal to play a checker for a remaining die
  // except when there are 2 different dice remaining to be played.
  // In this case, we must check if a single checker move is actually legal
  // by verifying the other die can be played from the resulting position.
  bool bNeedsVerification = dice_[0] != dice_[1]
      && remaining_dice_[dice_[0] - 1] > 0
      && remaining_dice_[dice_[1] - 1] > 0;

  if (!bNeedsVerification) {
    for (int die : dice_) {
      if (remaining_dice_[die - 1] > 0) {
        return SingleCheckerMoves(die);
      }
    }
    return {};
  }

  std::set<CheckerMove> moves;
  std::unique_ptr<State> cstate = this->Clone();
  BackgammonState* state = dynamic_cast<BackgammonState*>(cstate.get());
  for (int die : dice_) {
    if (remaining_dice_[die - 1] > 0) {
      std::set<CheckerMove> moves_here = SingleCheckerMoves(die);

      // For each candidate move, verify the other die can be played.
      int other_die = die == dice_[0] ? dice_[1] : dice_[0];
      for (const CheckerMove& move : moves_here) {
        state->ApplyCheckerMove(move);
        if (!state->SingleCheckerMoves(
                other_die, /* bFirstOnly=*/ true).empty()) {
          moves.insert(move);
        }
        state->UndoCheckerMove(move);
      }
    }
  }
  return moves;
}

std::set<CheckerMove> BackgammonState::SingleCheckerMoves(
    int die, bool bFirstOnly) const {
  int player = cur_player_;
  std::set<CheckerMove> moves;

  if (bar_[player] > 0) {
    // If there are any checkers on the bar, must move them out first.
    int pos = PositionFromBar(player, die);
    if (NumOppCheckers(player, pos) <= 1) {
      bool hit = NumOppCheckers(player, pos) == 1;
      moves.insert(CheckerMove(kBarPos, die, hit));
    }
    return moves;
  }

  // Regular board moves.
  bool all_in_home = AllInHome(player);
  for (int i = 0; i < kNumPoints; ++i) {
    if (board_[player][i] > 0) {
      int pos = PositionFrom(player, i, die);
      if (pos == kScorePos && all_in_home) {
        // Check whether a bear off move is legal.

        // It is ok to bear off if all the checkers are at home and the
        // point being used to move from exactly matches the distance from
        // just stepping off the board.
        if ((player == kXPlayerId && i + die == 24) ||
            (player == kOPlayerId && i - die == -1)) {
          moves.insert(CheckerMove(i, die, false));
          if (bFirstOnly) {
            return moves;
          }
        } else {
          // Otherwise, a die can only be used to move a checker off if
          // there are no checkers further than it in the player's home.
          if (i == FurthestCheckerInHome(player)) {
            moves.insert(CheckerMove(i, die, false));
            if (bFirstOnly) {
              return moves;
            }
          }
        }
      } else if (pos != kScorePos && NumOppCheckers(player, pos) <= 1) {
        // Regular move.
        bool hit = NumOppCheckers(player, pos) == 1;
        moves.insert(CheckerMove(i, die, hit));
        if (bFirstOnly) {
          return moves;
        }
      }
    }
  }
  return moves;
}

LegalLevel BackgammonState::DetermineLegalLevel() const {
  if (dice_.empty()) {
    SpielFatalError("DetermineLegalLevel called with empty dice.");
  }
  int hi_die = std::max(dice_[0], dice_[1]);
  int lo_die = std::min(dice_[0], dice_[1]);
  std::unique_ptr<State> cstate = this->Clone();
  BackgammonState* state = dynamic_cast<BackgammonState*>(cstate.get());
  int num_dice_used = 0;
  bool hi_die_used = false;
  if (hi_die == lo_die) {
    num_dice_used = NumMaxPlayableDies(state, {hi_die, hi_die, hi_die, hi_die});
    hi_die_used = true;
  } else {
    num_dice_used = NumMaxPlayableDies(state, {lo_die, hi_die});
    if (num_dice_used > 0) {
      // the helper func pops dice from the back, so high die would have been tried first.
      hi_die_used = true;
    }
    if (num_dice_used < 2) {
      // swap dice and try again
      num_dice_used =
          std::max(num_dice_used, NumMaxPlayableDies(state, {hi_die, lo_die}));
    }
  }
  switch (num_dice_used) {
    case 1: return (hi_die_used ? LegalLevel::kHighDie : LegalLevel::kLowDie);
    case 2: return LegalLevel::kTwoDice;
    case 3: return LegalLevel::kThreeDice;
    case 4: return LegalLevel::kFourDice;
    default: return LegalLevel::kNoDice;
  }
}

// Helper for DetermineLegalLevel().
// Caller relies on dice_to_play being popped from the back.
int BackgammonState::NumMaxPlayableDies(
    BackgammonState* state, std::vector<int> dice_to_play) const {
  if (dice_to_play.empty()) return 0;
  int die = dice_to_play.back();
  dice_to_play.pop_back();
  std::set<CheckerMove> moves_here = state->SingleCheckerMoves(die);
  int child_max = -1;
  for (const CheckerMove& move : moves_here) {
    state->ApplyCheckerMove(move);
    int child_val = NumMaxPlayableDies(state, dice_to_play);
    state->UndoCheckerMove(move);
    if (child_val == dice_to_play.size()) {
      return 1 + child_val;
    } else {
      child_max = std::max(child_max, child_val);
    }
  }
  return 1 + child_max;
}

bool BackgammonState::ApplyCheckerMove(const CheckerMove& move) {
  // Pass does nothing.
  if (move.pos < 0) {
    return false;
  }
  int player = cur_player_;

  // First, remove the checker.
  int next_pos = -1;
  if (move.pos == kBarPos) {
    bar_[player]--;
    next_pos = PositionFromBar(player, move.num);
  } else {
    board_[player][move.pos]--;
    next_pos = PositionFrom(player, move.pos, move.num);
  }

  remaining_dice_[move.num - 1]--;

  // Now add the checker (or score).
  if (next_pos == kScorePos) {
    scores_[player]++;
  } else {
    board_[player][next_pos]++;
  }

  bool hit = false;
  // If there was a hit, remove opponent's piece and add to bar.
  // Note: the move.hit will only be properly set during the legal moves search,
  // so we have to also check here if there is a hit candidate.
  if (move.hit ||
      (next_pos != kScorePos && board_[Opponent(player)][next_pos] == 1)) {
    hit = true;
    board_[Opponent(player)][next_pos]--;
    bar_[Opponent(player)]++;
  }

  return hit;
}

// Undoes a checker move. Important note: this checkermove needs to have
// move.hit set from the history to properly undo a move (this information is
// not tracked in the action value).
void BackgammonState::UndoCheckerMove(const CheckerMove& move) {
  // Undoing a pass does nothing
  if (move.pos < 0) {
    return;
  }

  int player = cur_player_;

  // First, figure out the next position.
  int next_pos = -1;
  if (move.pos == kBarPos) {
    next_pos = PositionFromBar(player, move.num);
  } else {
    next_pos = PositionFrom(player, move.pos, move.num);
  }

  // If there was a hit, take it out of the opponent's bar and put it back
  // onto the next position.
  if (move.hit) {
    bar_[Opponent(player)]--;
    board_[Opponent(player)][next_pos]++;
  }

  // Remove the moved checker or decrement score.
  if (next_pos == kScorePos) {
    scores_[player]--;
  } else {
    board_[player][next_pos]--;
  }
  remaining_dice_[move.num - 1]++;

  // Finally, return back the checker to its original position.
  if (move.pos == kBarPos) {
    bar_[player]++;
  } else {
    board_[player][move.pos]++;
  }
}

std::vector<Action> BackgammonState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};
  if (dice_.empty()) {
    return { kRollAction };
  }

  SPIEL_CHECK_EQ(CountTotalCheckers(kXPlayerId),
                 NumCheckersPerPlayer(game_.get()));
  SPIEL_CHECK_EQ(CountTotalCheckers(kOPlayerId),
                 NumCheckersPerPlayer(game_.get()));

  // Checker play
  std::vector<Action> legal_actions;
  if (kNumMovesPerCheckerSequence==1) {
    // Avoid the overhead of constructing move sequeneces.
    std::set<CheckerMove> moves = LegalSingleCheckerMoves();
    if (moves.empty()) {
      return { kEndTurnAction };
    }
    for (const CheckerMove& move : moves) {
      legal_actions.push_back(SingleCheckerMoveToSpielMove(move));
    }
  } else {
    std::set<CheckerMoveSequence> seqs = LegalCheckerMoveSequences();
    if (seqs.empty()) {
      return { kEndTurnAction };
    }
    for (const CheckerMoveSequence& seq : seqs) {
      Action action = CheckerMovesToSpielMove(seq.GetMoves());
      if (action > 0) {
        legal_actions.push_back(action);
      }
    }
  }
  std::sort(legal_actions.begin(), legal_actions.end());
  /*
  for (Action action : legal_actions) {
    if (action < 0 || action >= 155) {
      std::cerr << "Bad training inputs in LegalActions(): " << std::endl
          << "Legal actions: " << legal_actions << std::endl
          << "Observations: " << ObservationString(cur_player_) << std::endl;
    }
  }
  */
  return legal_actions;
}

std::vector<std::pair<Action, double>> BackgammonState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  return (turns_ == -1) ? kFirstRollChanceOutcomes : kChanceOutcomes;
}

std::string BackgammonState::ToString() const {
  std::string board_str =
    (cur_player_ == kXPlayerId)
      ? "+24-23-22-21-20-19--BAR-18-17-16-15-14-13-+\n"
      : "+-1--2--3--4--5--6--BAR--7--8--9-10-11-12-+\n";
  
  int pip_height = 5; // max checkers on a pip 
  for (int half = 0; half < 2; half++) {
    for (int i = 0; i < pip_height; i++) {
      absl::StrAppend(&board_str, "|");
      int pt_thresh = (half == 0) ? i : pip_height-i-1;
      int bar_thresh = (half == 1) ? i+1 : pip_height-i;
      for (int j = 0; j < 12; j++) {
        int pt = (half == 0) ? j : 23-j;
        if (pt == 6) {
          //add the bar for X's
          absl::StrAppend(&board_str, "|");
          int num = bar_[kXPlayerId];
          if (i > 0 && num > bar_thresh) {
            if (i == 1 && num > pip_height) {
              absl::StrAppend(&board_str, absl::StrFormat("%2d ", num));
            } else {
              absl::StrAppend(&board_str, " X ");
            }
          } else {
            absl::StrAppend(&board_str, "   ");
          }
          absl::StrAppend(&board_str, "|");
        }
        if (pt == 17) {
          //add the bar for O's
          absl::StrAppend(&board_str, "|");

          int num = bar_[kOPlayerId];
          if (i < pip_height - 1 && num > bar_thresh) {
            if (i == pip_height - 2 && num > pip_height) {
              absl::StrAppend(&board_str, absl::StrFormat("%2d ", num));
            }
            else {
              absl::StrAppend(&board_str, " O ");
            }
          }
          else {
            absl::StrAppend(&board_str, "   ");
          }
          absl::StrAppend(&board_str, "|");
        }

        int numX = board_[kXPlayerId][pt];
        int numO = board_[kOPlayerId][pt];
        if (numX > pt_thresh || numO > pt_thresh) {
          int num = std::max(numX, numO);
          if (i == (pip_height - 1) * (1 - half) && num > pip_height) {
              absl::StrAppend(&board_str, absl::StrFormat("%2d ", num));
          } else {
              absl::StrAppend(&board_str, numX > 0 ? " X " : " O ");
          }
        } else {
          absl::StrAppend(&board_str, "   ");
        }
      }
      absl::StrAppend(&board_str, "|");

      //add player info
      if (i == 0 && half == 0) {
        absl::StrAppend(&board_str, " Player O");
        absl::StrAppend(&board_str, absl::StrFormat("  Off: %d", scores_[kOPlayerId]));
      }
      if (i == 4 && half == 1) {
        absl::StrAppend(&board_str, " Player X");
        absl::StrAppend(&board_str, absl::StrFormat("  Off: %d", scores_[kXPlayerId]));
      }      
      absl::StrAppend(&board_str, "\n");
    }

    if (half == 0) {
      //do the middle
      absl::StrAppend(&board_str, "|                  |");
      absl::StrAppend(&board_str, (bar_[kXPlayerId] > 0) ? " X " : "   ");
      absl::StrAppend(&board_str,"|                  | ");

      switch (scoring_type_) {
        case ScoringType::kWinLossScoring:
          absl::StrAppend(&board_str, "1 Pt Match");
          break;
        case ScoringType::kEnableGammons:
          absl::StrAppend(&board_str, "Gammons Enabled");
          break;
        case ScoringType::kFullScoring:
          absl::StrAppend(&board_str, "Full Scoring");
          break;
        default:
          break;
      }
      absl::StrAppend(&board_str, "\n|                  |");
      absl::StrAppend(&board_str, (bar_[kOPlayerId] > 0) ? " O " : "   ");
      absl::StrAppend(&board_str,"|                  | ");
      absl::StrAppend(&board_str, absl::StrFormat("Turn: %s  Dice:", CurPlayerToString(cur_player_)));
      /*
      if (!dice_.empty()) {
        absl::StrAppend(&board_str,
            absl::StrFormat(" %s %s", DiceToString(dice_[0]), DiceToString(dice_[1])));
      */
      if (!remaining_dice_.empty()) {
        for (int die = 0; die < kNumDiceOutcomes; die++) {
          for (int i = 0; i < remaining_dice_[die]; i++) {
            absl::StrAppend(&board_str, absl::StrFormat(" %s", DiceToString(die + 1)));
          }
        }
      }
      absl::StrAppend(&board_str, "\n");
    }
  }
  absl::StrAppend(&board_str, 
    (cur_player_ == kXPlayerId)
      ? "+-1--2--3--4--5--6--BAR--7--8--9-10-11-12-+\n"
      : "+24-23-22-21-20-19--BAR-18-17-16-15-14-13-+\n");
  absl::StrAppend(&board_str,
      absl::StrFormat("PositionID: %s \n", PositionId()));
    
  return board_str;
}

std::string BackgammonState::PositionId() const {
  // TODO:  use xg format
  char val;
  std::string retval;
  for (int i = 0; i < 24; i++) {
    if (board_[kXPlayerId][i] > 0)
      val = 'A' + board_[kXPlayerId][i] - 1;
    else if (board_[kOPlayerId][i] > 0)
      val = 'a' + board_[kOPlayerId][i] - 1;
    else
      val = '-';
    retval += val;
  }
  retval += 'A' + bar_[kXPlayerId];
  retval += 'a' + bar_[kOPlayerId];

  return retval;
}

bool BackgammonState::IsTerminal() const {
  return (scores_[kXPlayerId] == NumCheckersPerPlayer(game_.get()) ||
          scores_[kOPlayerId] == NumCheckersPerPlayer(game_.get()));
}

std::vector<double> BackgammonState::Returns() const {
  int winner = -1;
  int loser = -1;
  if (scores_[kXPlayerId] == 15) {
    winner = kXPlayerId;
    loser = kOPlayerId;
  } else if (scores_[kOPlayerId] == 15) {
    winner = kOPlayerId;
    loser = kXPlayerId;
  } else {
    return {0.0, 0.0};
  }

  // Magnify the util based on the scoring rules for this game.
  int util_mag = 1;
  switch (scoring_type_) {
    case ScoringType::kWinLossScoring:
    default:
      break;

    case ScoringType::kEnableGammons:
      util_mag = (IsGammoned(loser) ? 2 : 1);
      break;

    case ScoringType::kFullScoring:
      util_mag = (IsBackgammoned(loser) ? 3 : IsGammoned(loser) ? 2 : 1);
      break;
  }

  std::vector<double> returns(kNumPlayers);
  returns[winner] = util_mag;
  returns[loser] = -util_mag;
  return returns;
}

std::unique_ptr<State> BackgammonState::Clone() const {
  return std::unique_ptr<State>(new BackgammonState(*this));
}

void BackgammonState::SetState(int cur_player, bool double_turn,
                               const std::vector<int>& dice,
                               const std::vector<int>& bar,
                               const std::vector<int>& scores,
                               const std::vector<std::vector<int>>& board) {
  cur_player_ = cur_player;
  double_turn_ = double_turn;
  dice_ = dice;
  bar_ = bar;
  scores_ = scores;
  board_ = board;
  InitRemainingDice();

  SPIEL_CHECK_EQ(CountTotalCheckers(kXPlayerId),
                 NumCheckersPerPlayer(game_.get()));
  SPIEL_CHECK_EQ(CountTotalCheckers(kOPlayerId),
                 NumCheckersPerPlayer(game_.get()));
}

BackgammonGame::BackgammonGame(const GameParameters& params)
    : Game(kGameType, params),
      scoring_type_(
          ParseScoringType(ParameterValue<std::string>("scoring_type"))),
      hyper_backgammon_(ParameterValue<bool>("hyper_backgammon")) {}

double BackgammonGame::MaxUtility() const {
  if (hyper_backgammon_) {
    // We do not have the cube implemented, so Hyper-backgammon us currently
    // restricted to a win-loss game regardless of the scoring type.
    return 1;
  }

  switch (scoring_type_) {
    case ScoringType::kWinLossScoring:
      return 1;
    case ScoringType::kEnableGammons:
      return 2;
    case ScoringType::kFullScoring:
      return 3;
    default:
      SpielFatalError("Unknown scoring_type");
  }
}

int BackgammonGame::NumCheckersPerPlayer() const {
  if (hyper_backgammon_) {
    return 3;
  } else {
    return kNumCheckersPerPlayer;
  }
}

CheckerMoveSequence::CheckerMoveSequence(const BackgammonState& state) :
    state_(state) {}

void CheckerMoveSequence::AddMove(const CheckerMove& move) {
  moves_.push_back(move);
  state_.ApplyCheckerMove(move);
  id_ = state_.PositionId();
}
}  // namespace backgammon
}  // namespace open_spiel
