// Copyright 2021 DeepMind Technologies Limited
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

#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/bots/human/human_bot.h"
#include "open_spiel/games/backgammon.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, game, "tic_tac_toe", "The name of the game to play.");
ABSL_FLAG(std::string, player1, "az", "Who controls player1.");
ABSL_FLAG(std::string, player2, "random", "Who controls player2.");
ABSL_FLAG(std::string, az_path, "", "Path to AZ experiment.");
ABSL_FLAG(std::string, az_graph_def, "vpnet.pb",
          "AZ graph definition file name.");
ABSL_FLAG(double, uct_c, 2, "UCT exploration constant.");
ABSL_FLAG(int, rollout_count, 10, "How many rollouts per evaluation.");
ABSL_FLAG(int, min_simulations, 0, "How many simulations to run (min).");
ABSL_FLAG(int, max_simulations, 10000, "How many simulations to run (max).");
ABSL_FLAG(int, num_games, 1, "How many games to play.");
ABSL_FLAG(int, max_memory_mb, 1000,
          "The maximum memory used before cutting the search short.");
ABSL_FLAG(int, az_checkpoint, -1, "Checkpoint of AZ model.");
ABSL_FLAG(int, az_batch_size, 1, "Batch size of AZ inference.");
ABSL_FLAG(int, az_threads, 1, "Number of threads to run for AZ inference.");
ABSL_FLAG(int, az_cache_size, 16384, "Cache size of AZ algorithm.");
ABSL_FLAG(int, az_cache_shards, 1, "Cache shards of AZ algorithm.");
ABSL_FLAG(bool, solve, true, "Whether to use MCTS-Solver.");
ABSL_FLAG(uint_fast32_t, seed, 0, "Seed for MCTS.");
ABSL_FLAG(bool, verbose, false, "Show the MCTS stats of possible moves.");
ABSL_FLAG(bool, quiet, false, "Show the MCTS stats of possible moves.");

uint_fast32_t Seed() {
  uint_fast32_t seed = absl::GetFlag(FLAGS_seed);
  return seed != 0 ? seed : absl::ToUnixMicros(absl::Now());
}

std::unique_ptr<open_spiel::Bot>
InitBot(std::string type, const open_spiel::Game &game,
        open_spiel::Player player,
        std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator,
        std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator>
            az_evaluator) {
  if (type == "az") {
    return std::make_unique<open_spiel::algorithms::MCTSBot>(
        game, std::move(az_evaluator), absl::GetFlag(FLAGS_uct_c),
        absl::GetFlag(FLAGS_min_simulations),
        absl::GetFlag(FLAGS_max_simulations),
        absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), Seed(),
        absl::GetFlag(FLAGS_verbose),
        open_spiel::algorithms::ChildSelectionPolicy::PUCT, 0, 0,
        /*dont_return_chance_node=*/true);
  }
  if (type == "human") {
    return std::make_unique<open_spiel::HumanBot>();
  }
  if (type == "mcts") {
    return std::make_unique<open_spiel::algorithms::MCTSBot>(
        game, std::move(evaluator), absl::GetFlag(FLAGS_uct_c),
        absl::GetFlag(FLAGS_min_simulations),
        absl::GetFlag(FLAGS_max_simulations),
        absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), Seed(),
        absl::GetFlag(FLAGS_verbose));
  }
  if (type == "random") {
    return open_spiel::MakeUniformRandomBot(player, Seed());
  }

  open_spiel::SpielFatalError(
      "Bad player type. Known types: az, human, mcts, random");
}

open_spiel::Action GetAction(const open_spiel::State &state,
                             std::string action_str) {
  for (open_spiel::Action action : state.LegalActions()) {
    if (action_str == state.ActionToString(state.CurrentPlayer(), action))
      return action;
  }
  return open_spiel::kInvalidAction;
}

void ExportBackgammonMatFile(
    const open_spiel::Game& game,
    const std::vector<open_spiel::Action>& actions) {

  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  auto bstate =
      dynamic_cast<open_spiel::backgammon::BackgammonState*>(state.get());

  //match length info
  std::cerr << "1 Point Match" << std::endl << std::endl;
  std::cerr << " Game 1" << std::endl;
  std::cerr << absl::StrFormat(" %-31s %-31s", "ZiggyX : 0", "ZiggyO : 0");
  std::cerr << std::endl;

  int movenum = 0;
  bool newline = false;
  std::string move_str = "";

  for (open_spiel::Action action : actions) {
    int player = bstate->CurrentPlayer();
    if (movenum == 0) {
      if (player >= 0) {
        movenum++;
        std::cerr << absl::StrFormat("%3d)", movenum);
        // add spacing based on which player is starting the game
        if (player == open_spiel::backgammon::kOPlayerId) {
          std::cerr << absl::StrFormat("%32s", "");
          newline = true;
        }
      }
    } else if (player < 0) {
      // end turn
      std::cerr << absl::StrFormat(" %-31s", move_str);
      move_str = "";
      if (newline) {
        movenum++;
        std::cerr << std::endl << absl::StrFormat("%3d)", movenum);
      }
      newline = !newline;
    }
    move_str = absl::StrCat(move_str, bstate->ActionToMatString(action));
    bstate->ApplyAction(action);
  }
  std::cerr << absl::StrFormat(" %-31s", move_str) << std::endl;
  std::string win_str = "Wins 1 point and the match";
  std::cerr << absl::StrFormat(" %-31s %s",
      newline ? "" : win_str,
      newline ? win_str : "") << std::endl;
}

double EvaluateLuck(
    const open_spiel::State& state,
    open_spiel::Action action,
    std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator) {
  SPIEL_CHECK_TRUE(state.IsChanceNode());
  double avg_value = 0;
  double action_value = 0;
  bool found = false;
  for (auto& pair : state.ChanceOutcomes()) {
    std::unique_ptr<open_spiel::State> temp_state = state.Clone();
    open_spiel::Action temp_action = pair.first;
    double temp_prob = pair.second;
    temp_state->ApplyAction(temp_action);
    double temp_value = evaluator->Evaluate(*temp_state)[0];
    if (temp_action == action) {
      found = true;
      action_value = temp_value;
    }
    avg_value += temp_prob * temp_value;
  }
  SPIEL_CHECK_TRUE(found);
  return action_value - avg_value;
}

std::pair<std::vector<double>, std::string>
PlayGame(const open_spiel::Game &game,
         std::vector<std::unique_ptr<open_spiel::Bot>> &bots,
         std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator,
         std::mt19937 &rng, const std::vector<std::string> &initial_actions) {
  bool quiet = absl::GetFlag(FLAGS_quiet);
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::vector<std::string> history;
  std::vector<open_spiel::Action> actions;

  if (!quiet)
    std::cerr << "Initial state:\n" << state << std::endl;

  // Play the initial actions (if there are any).
  for (const auto &action_str : initial_actions) {
    open_spiel::Player current_player = state->CurrentPlayer();
    open_spiel::Action action = GetAction(*state, action_str);

    if (action == open_spiel::kInvalidAction)
      open_spiel::SpielFatalError(absl::StrCat("Invalid action: ", action_str));

    history.push_back(action_str);
    actions.push_back(action);
    state->ApplyAction(action);

    if (!quiet) {
      std::cerr << "Player " << current_player
                << " forced action: " << action_str << std::endl;
      std::cerr << "Next state:\n" << state->ToString() << std::endl;
    }
  }

  double accum_luck = 0;
  while (!state->IsTerminal()) {
    /*
    if (!state->IsChanceNode()) {
      for (auto v : state->ObservationTensor()) {
        std::cerr << v << " ";
      }
      std::cerr << std::endl;
    }
    */
    open_spiel::Player player = state->CurrentPlayer();

    open_spiel::Action action;
    bool was_chance = false;
    double luck = 0;
    if (state->IsChanceNode()) {
      // Chance node; sample one according to underlying distribution.
      open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
      action = open_spiel::SampleAction(outcomes, rng).first;
      was_chance = true;
    } else {
      // The state must be a decision node, ask the right bot to make its
      // action.
      action = bots[player]->Step(*state);
    }
    if (!quiet) {
      std::cerr << "Player " << player
                << " chose action: " << state->ActionToString(player, action)
                << std::endl;

      if (was_chance) {
        luck = EvaluateLuck(*state, action, evaluator);
      }
    }

    // Inform the other bot of the action performed.
    for (open_spiel::Player p = 0; p < bots.size(); ++p) {
      if (p != player) {
        bots[p]->InformAction(*state, player, action);
      }
    }

    // Update history and get the next state.
    history.push_back(state->ActionToString(player, action));
    actions.push_back(action);
    state->ApplyAction(action);

    if (!quiet && (was_chance || absl::GetFlag(FLAGS_game)!="backgammon")) {
      // mattrek: for backgammon: only print new board after dice rolls
      accum_luck += luck;
      luck *= (state->CurrentPlayer() == 0 ? 1 : -1);
      std::cerr << "Luck: " << luck << "  Accum Luck: " << accum_luck << std::endl;
      std::cerr << "Next state:\n" << state->ToString() << std::endl;
    }
  }

  std::string history_str = absl::StrJoin(history, ", ");
  std::cerr << "Returns: " << absl::StrJoin(state->Returns(), ", ")
            << std::endl;
  std::cerr << "Game actions: " << history_str << std::endl;

  if (absl::GetFlag(FLAGS_game)=="backgammon") {
    ExportBackgammonMatFile(game, actions);
  }
  return {state->Returns(), history_str};
}

int main(int argc, char **argv) {
  std::vector<char *> positional_args = absl::ParseCommandLine(argc, argv);
  std::mt19937 rng(Seed());  // Random number generator.

  // Create the game.
  std::string game_name = absl::GetFlag(FLAGS_game);
  std::cerr << "Game: " << game_name << std::endl;
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(game_name);

  // Ensure the game is AlphaZero-compatible and arguments are compatible.
  open_spiel::GameType game_type = game->GetType();
  if (game->NumPlayers() != 2)
    open_spiel::SpielFatalError("AlphaZero can only handle 2-player games.");
  if (game_type.reward_model != open_spiel::GameType::RewardModel::kTerminal)
    open_spiel::SpielFatalError("Game must have terminal rewards.");
  if (game_type.dynamics != open_spiel::GameType::Dynamics::kSequential)
    open_spiel::SpielFatalError("Game must have sequential turns.");
  if (absl::GetFlag(FLAGS_az_path).empty())
    open_spiel::SpielFatalError("AlphaZero path must be specified.");
  if (absl::GetFlag(FLAGS_player1) != "az" &&
      absl::GetFlag(FLAGS_player2) != "az")
    open_spiel::SpielFatalError("One of the players must be AlphaZero.");

  open_spiel::algorithms::torch_az::DeviceManager device_manager;
  device_manager.AddDevice(open_spiel::algorithms::torch_az::VPNetModel(
      *game, absl::GetFlag(FLAGS_az_path), absl::GetFlag(FLAGS_az_graph_def),
      "/cpu:0"));
  device_manager.Get(0, 0)->LoadCheckpoint(absl::GetFlag(FLAGS_az_checkpoint));
  device_manager.Get(0, 0)->Print();
  auto az_evaluator =
      std::make_shared<open_spiel::algorithms::torch_az::VPNetEvaluator>(
          /*device_manager=*/&device_manager,
          /*batch_size=*/absl::GetFlag(FLAGS_az_batch_size),
          /*threads=*/absl::GetFlag(FLAGS_az_threads),
          /*cache_size=*/absl::GetFlag(FLAGS_az_cache_size),
          /*cache_shards=*/absl::GetFlag(FLAGS_az_cache_shards));
  auto evaluator =
      std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
          absl::GetFlag(FLAGS_rollout_count), Seed());

  std::vector<std::unique_ptr<open_spiel::Bot>> bots;
  bots.push_back(
      InitBot(absl::GetFlag(FLAGS_player1), *game, 0, evaluator, az_evaluator));
  bots.push_back(
      InitBot(absl::GetFlag(FLAGS_player2), *game, 1, evaluator, az_evaluator));

  std::vector<std::string> initial_actions;
  for (int i = 1; i < positional_args.size(); ++i) {
    initial_actions.push_back(positional_args[i]);
  }

  std::map<std::string, int> histories;
  std::vector<double> overall_returns(2, 0);
  std::vector<int> overall_wins(2, 0);
  int num_games = absl::GetFlag(FLAGS_num_games);
  for (int game_num = 0; game_num < num_games; ++game_num) {
    auto [returns, history_str] =
        PlayGame(*game, bots, az_evaluator, rng, initial_actions);
    histories[history_str] += 1;
    for (int i = 0; i < returns.size(); ++i) {
      double v = returns[i];
      overall_returns[i] += v;
      if (v > 0) {
        overall_wins[i] += 1;
      }
    }
  }

  std::cerr << "Number of games played: " << num_games << std::endl;
  std::cerr << "Number of distinct games played: " << histories.size()
            << std::endl;
  std::cerr << "Players: " << absl::GetFlag(FLAGS_player1) << ", "
            << absl::GetFlag(FLAGS_player2) << std::endl;
  std::cerr << "Overall wins: " << absl::StrJoin(overall_wins, ", ")
            << std::endl;
  std::cerr << "Overall returns: " << absl::StrJoin(overall_returns, ", ")
            << std::endl;

  return 0;
}
