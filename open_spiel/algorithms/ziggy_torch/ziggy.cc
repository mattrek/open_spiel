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

#include "open_spiel/algorithms/ziggy_torch/ziggy.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/random/uniform_real_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/algorithms/ziggy_torch/device_manager.h"
#include "open_spiel/algorithms/ziggy_torch/v_evaluator.h"
#include "open_spiel/algorithms/ziggy_torch/v_net.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/circular_buffer.h"
#include "open_spiel/utils/data_logger.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/logger.h"
#include "open_spiel/utils/lru_cache.h"
#include "open_spiel/utils/serializable_circular_buffer.h"
#include "open_spiel/utils/stats.h"
#include "open_spiel/utils/thread.h"
#include "open_spiel/utils/threaded_queue.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

struct StartInfo {
  absl::Time start_time;
  int start_step;
  int model_checkpoint_step;
  int64_t total_trajectories;
};

StartInfo StartInfoFromLearnerJson(const std::string& path) {
  StartInfo start_info;
  file::File learner_file(path + "/learner.jsonl", "r");
  std::vector<std::string> learner_lines = absl::StrSplit(
      learner_file.ReadContents(), "\n");
  std::string last_learner_line;

  // Get the last non-empty line in learner.jsonl.
  for (int i = learner_lines.size() - 1; i >= 0; i--) {
    if (!learner_lines[i].empty()) {
      last_learner_line = learner_lines[i];
      break;
    }
  }

  json::Object last_learner_json = json::FromString(
      last_learner_line).value().GetObject();

  start_info.start_time = absl::Now() - absl::Seconds(
      last_learner_json["time_rel"].GetDouble());
  start_info.start_step = last_learner_json["step"].GetInt() + 1;
  start_info.model_checkpoint_step = VPNetModel::kMostRecentCheckpointStep;
  start_info.total_trajectories =
      last_learner_json["total_trajectories"].GetInt();

  return start_info;
}

struct Trajectory {
  struct State {
    std::vector<float> observation;
    open_spiel::Player current_player;
    open_spiel::Action action;
    double value; // value after applying action
    double accum_luck;  // always for player 0 (i.e. not current_player)
  };

  std::vector<State> states;
  std::vector<double> returns;
};

double EvaluateLuck(
    const open_spiel::State& state,
    open_spiel::Action action,
    std::shared_ptr<Evaluator> evaluator) {
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
      SPIEL_CHECK_FALSE(found);
      found = true;
      action_value = temp_value;
    }
    avg_value += temp_prob * temp_value;
  }
  SPIEL_CHECK_TRUE(found);
  return action_value - avg_value;
}

Trajectory PlayGame(Logger* logger, int game_num, const open_spiel::Game& game,
                    std::vector<std::unique_ptr<MCTSBot>>* bots,
                    std::shared_ptr<Evaluator> evaluator,
                    std::mt19937* rng, double temperature, int temperature_drop,
                    double cutoff_value, bool verbose = true) {
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::vector<std::string> history;
  Trajectory trajectory;
  double accum_luck = 0;

  while (true) {
    if (state->IsChanceNode()) {
      open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
      open_spiel::Action action =
          open_spiel::SampleAction(outcomes, *rng).first;
      accum_luck += EvaluateLuck(*state, action, evaluator);
      state->ApplyAction(action);
    } else {
      open_spiel::Player player = state->CurrentPlayer();
      std::unique_ptr<SearchNode> root = (*bots)[player]->MCTSearch(*state);
      const SearchNode* action_node;
      if (history.size() >= temperature_drop) {
        action_node = &(root->BestChild());
      } else {
        // mattrek: TODO add SampleFromChildren func, reuse in mcts.cc
        open_spiel::SpielFatalError("Need SampleFromChildren to use temp drop.");
        /*
        open_spiel::Action action;
        action = open_spiel::SampleAction(policy, *rng).first;
        for (const SearchNode& child : root->children) {
          if (child.action == action) {
            action_node = &child;
            break;
          }
        }
        */
      }

      // mattrek: Assert best_child.player == player, since this should always
      // be the case given how MCTSearch expands root node's children.
      SPIEL_CHECK_TRUE(action_node->player == player);
      // mattrek: When adding this state into the trajectory, assign it the
      // value of the chosen action.  Do not use the MCTS search value for this,
      // use the vpnet's evaluation.  (The MCTS value is distorted from uct_c
      // exploration).  An exception is made for chance nodes, as the vpnet
      // cannot evaluate them, and their children are visited independent
      // of uct_c.
      double action_value =
          action_node->outcome.empty()
              ? action_node->eval
              : action_node->outcome[player];
      trajectory.states.push_back(Trajectory::State{
          state->ObservationTensor(), player,
          action_node->action, action_value, accum_luck});
      std::string action_str =
          state->ActionToString(player, action_node->action);
      history.push_back(action_str);
      state->ApplyAction(action_node->action);
      if (verbose) {
        logger->Print("Player: %d, action: %s, value: %6.3f, accum_luck: %6.3f",
            player, action_str, action_value, accum_luck);
      }
      if (state->IsTerminal()) {
        trajectory.returns = state->Returns();
        break;
      } else if (std::abs(action_value) > cutoff_value) {
        trajectory.returns.resize(2);
        trajectory.returns[player] = action_value;
        trajectory.returns[1 - player] = -action_value;
        break;
      }
    }
  }

  logger->Print("Game %d: Returns: %s; Actions: %s", game_num,
                absl::StrJoin(trajectory.returns, " "),
                absl::StrJoin(history, " "));
  return trajectory;
}

std::unique_ptr<MCTSBot> InitAZBot(const AlphaZeroConfig& config,
                                   const open_spiel::Game& game,
                                   std::shared_ptr<Evaluator> evaluator,
                                   bool evaluation) {
  return std::make_unique<MCTSBot>(
      game, std::move(evaluator), config.uct_c,
      config.min_simulations, config.max_simulations,
      /*max_memory_mb=*/10,
      /*solve=*/false,
      /*seed=*/0,
      /*verbose=*/true, ChildSelectionPolicy::PUCT,
      evaluation ? 0 : config.policy_alpha,
      evaluation ? 0 : config.policy_epsilon,
      /*dont_return_chance_node*/ true);
}

// An actor thread runner that generates games and returns trajectories.
void actor(const open_spiel::Game& game, const AlphaZeroConfig& config, int num,
           ThreadedQueue<Trajectory>* trajectory_queue,
           std::shared_ptr<VPNetEvaluator> vp_eval, StopToken* stop) {
  std::unique_ptr<Logger> logger;
  if (num < 20) {  // Limit the number of open files.
    logger.reset(new FileLogger(config.path, absl::StrCat("actor-", num)));
  } else {
    logger.reset(new NoopLogger());
  }
  std::mt19937 rng(absl::ToUnixNanos(absl::Now()));
  absl::uniform_real_distribution<double> dist(0.0, 1.0);
  std::vector<std::unique_ptr<MCTSBot>> bots;
  bots.reserve(2);
  for (int player = 0; player < 2; player++) {
    bots.push_back(InitAZBot(config, game, vp_eval, /*evaluation=*/false));
  }
  for (int game_num = 1; !stop->StopRequested(); ++game_num) {
    double cutoff =
        (dist(rng) < config.cutoff_probability ? config.cutoff_value
                                               : game.MaxUtility() + 1);
    if (!trajectory_queue->Push(
            PlayGame(logger.get(), game_num, game, &bots, vp_eval, &rng,
                     config.temperature, config.temperature_drop, cutoff),
            absl::Seconds(10))) {
      logger->Print("Failed to push a trajectory after 10 seconds.");
    }
  }
  logger->Print("Got a quit.");
}

class EvalResults {
 public:
  explicit EvalResults(int count, int evaluation_window) {
    results_.reserve(count);
    for (int i = 0; i < count; ++i) {
      results_.emplace_back(evaluation_window);
    }
  }

  // How many evals per difficulty.
  int EvalCount() {
    absl::MutexLock lock(&m_);
    return eval_num_ / results_.size();
  }

  // Which eval to do next: difficulty, player0.
  std::pair<int, bool> Next() {
    absl::MutexLock lock(&m_);
    int next = eval_num_ % (results_.size() * 2);
    eval_num_ += 1;
    return {next / 2, next % 2};
  }

  void Add(int i, double value) {
    absl::MutexLock lock(&m_);
    results_[i].Add(value);
  }

  std::vector<double> AvgResults() {
    absl::MutexLock lock(&m_);
    std::vector<double> out;
    out.reserve(results_.size());
    for (const auto& result : results_) {
      out.push_back(result.Empty() ? 0
                                   : (absl::c_accumulate(result.Data(), 0.0) /
                                      result.Size()));
    }
    return out;
  }

 private:
  std::vector<CircularBuffer<double>> results_;
  int eval_num_ = 0;
  absl::Mutex m_;
};

// A thread that plays vs standard MCTS.
void evaluator(const open_spiel::Game& game, const AlphaZeroConfig& config,
               int num, EvalResults* results,
               std::shared_ptr<VPNetEvaluator> vp_eval, StopToken* stop) {
  FileLogger logger(config.path, absl::StrCat("evaluator-", num));
  std::mt19937 rng;
  auto rand_evaluator = std::make_shared<RandomRolloutEvaluator>(1, num);

  for (int game_num = 1; !stop->StopRequested(); ++game_num) {
    auto [difficulty, first] = results->Next();
    int az_player = first ? 0 : 1;
    int rand_max_simulations =
        config.max_simulations * std::pow(10, difficulty / 2.0);
    std::vector<std::unique_ptr<MCTSBot>> bots;
    bots.reserve(2);
    bots.push_back(InitAZBot(config, game, vp_eval, /*evaluation=*/true));
    bots.push_back(std::make_unique<MCTSBot>(
        game, rand_evaluator, config.uct_c,
        /*min_simulations=*/0, rand_max_simulations,
        /*max_memory_mb=*/1000,
        /*solve=*/true,
        /*seed=*/num * 1000 + game_num,
        /*verbose=*/false, ChildSelectionPolicy::UCT,
        /*dirichlet_alpha=*/0,
        /*dirichlet_epsilon=*/0,
        /*dont_return_chance_node=*/true));
    if (az_player == 1) {
      std::swap(bots[0], bots[1]);
    }

    logger.Print("Running MCTS with %d simulations", rand_max_simulations);
    Trajectory trajectory = PlayGame(
        &logger, game_num, game, &bots, vp_eval, &rng, /*temperature=*/1,
        /*temperature_drop=*/0, /*cutoff_value=*/game.MaxUtility() + 1);

    results->Add(difficulty, trajectory.returns[az_player]);
    logger.Print("Game %d: AZ: %5.2f, MCTS: %5.2f, MCTS-sims: %d, length: %d",
                 game_num, trajectory.returns[az_player],
                 trajectory.returns[1 - az_player], rand_max_simulations,
                 trajectory.states.size());
  }
  logger.Print("Got a quit.");
}

// Returns the 'lambda' discounted value of all future values of 'trajectory',
// including its outcome, beginning at 'state_idx'.  The calculation is
// truncated after 'td_n_steps' if that parameter is greater than zero.
double TdLambdaReturns(const Trajectory& trajectory, int state_idx,
                       double td_lambda, int td_n_steps) {
  const Trajectory::State& s_state = trajectory.states[state_idx];
  double accum_luck = s_state.accum_luck;
  double outcome = trajectory.returns[0]
      - (trajectory.states.back().accum_luck - accum_luck);
  outcome = std::max(-1.0, std::min(1.0, outcome));
  if (td_lambda >= 1.0 || Near(td_lambda, 1.0)) {
    // lambda == 1.0 simplifies to returning the outcome (or value at nth-step)
    if (td_n_steps <= 0) {
      return outcome;
    }
    int idx = state_idx + td_n_steps;
    if (idx >= trajectory.states.size()) {
      return outcome;
    }
    const Trajectory::State& n_state = trajectory.states[idx];
    return n_state.value * (n_state.current_player == 0 ? 1 : -1)
        - (n_state.accum_luck - accum_luck);
  }
  double retval = s_state.value * (s_state.current_player == 0 ? 1 : -1);
  if (td_lambda <= 0.0 || Near(td_lambda, 0.0)) {
    // lambda == 0 simplifies to returning the start state's value
    return retval;
  }
  double lambda_inv = (1.0 - td_lambda);
  double lambda_pow = td_lambda;
  retval *= lambda_inv;
  for (int i = state_idx + 1; i < trajectory.states.size(); ++i) {
    const Trajectory::State& i_state = trajectory.states[i];
    double value = i_state.value * (i_state.current_player == 0 ? 1 : -1)
        - (i_state.accum_luck - accum_luck);
    if (td_n_steps > 0 && i == state_idx + td_n_steps) {
      retval += lambda_pow * value;
      return retval;
    }
    retval += lambda_inv * lambda_pow * value;
    lambda_pow *= td_lambda;
  }
  retval += lambda_pow * outcome;
  return retval;
}

void learner(const open_spiel::Game& game, const AlphaZeroConfig& config,
             DeviceManager* device_manager,
             DeviceManager* cpu_device_manager,
             std::shared_ptr<VPNetEvaluator> eval,
             ThreadedQueue<Trajectory>* trajectory_queue,
             EvalResults* eval_results, StopToken* stop,
             const StartInfo& start_info,
             bool verbose = false) {
  FileLogger logger(config.path, "learner", "a");
  DataLoggerJsonLines data_logger(
      config.path, "learner", true, "a", start_info.start_time);
  std::mt19937 rng;

  int device_id = 0;  // Do not change, the first device is the learner.
  logger.Print("Running the learner on device %d: %s", device_id,
               device_manager->Get(0, device_id)->Device());

  SerializableCircularBuffer<VPNetModel::TrainInputs> replay_buffer(
      config.replay_buffer_size);
  if (start_info.start_step > 1) {
    replay_buffer.LoadBuffer(config.path + "/replay_buffer.data");
  }
  int learn_rate = config.replay_buffer_size / config.replay_buffer_reuse;
  int64_t total_trajectories = start_info.total_trajectories;

  const int stage_count = 7;
  std::vector<open_spiel::BasicStats> value_accuracies(stage_count);
  std::vector<open_spiel::BasicStats> value_predictions(stage_count);
  open_spiel::BasicStats game_lengths;
  open_spiel::HistogramNumbered game_lengths_hist(game.MaxGameLength() + 1);

  open_spiel::HistogramNamed outcomes({"Player1", "Player2", "Draw"});
  // Actor threads have likely been contributing for a while, so put `last` in
  // the past to avoid a giant spike on the first step.
  absl::Time last = absl::Now() - absl::Seconds(60);
  for (int step = start_info.start_step;
       !stop->StopRequested() &&
           (config.max_steps == 0 || step <= config.max_steps);
       ++step) {
    outcomes.Reset();
    game_lengths.Reset();
    game_lengths_hist.Reset();
    for (auto& value_accuracy : value_accuracies) {
      value_accuracy.Reset();
    }
    for (auto& value_prediction : value_predictions) {
      value_prediction.Reset();
    }

    // Collect trajectories
    int queue_size = trajectory_queue->Size();
    int num_states = 0;
    int num_trajectories = 0;
    while (!stop->StopRequested() && num_states < learn_rate) {
      absl::optional<Trajectory> trajectory = trajectory_queue->Pop();
      if (trajectory) {
        num_trajectories += 1;
        total_trajectories += 1;
        game_lengths.Add(trajectory->states.size());
        game_lengths_hist.Add(trajectory->states.size());

        double p1_outcome = trajectory->returns[0];
        outcomes.Add(p1_outcome > 0 ? 0 : (p1_outcome < 0 ? 1 : 2));

        for (int i = 0; i < trajectory->states.size(); ++i ) {
          const Trajectory::State& state = trajectory->states[i];
          double value = TdLambdaReturns(*trajectory, i,
                                         config.td_lambda, config.td_n_steps);
          value *= (state.current_player == 0
                      || !open_spiel::kPlayerCentricObs) ? 1 : -1;

          replay_buffer.Add(VPNetModel::TrainInputs{state.observation, value});
          if (verbose && num_trajectories == 1) {
            double v0 = state.value * (state.current_player == 0 ? 1 : -1);
            logger.Print("Idx: %d  Player: %d  Value0: %0.3f  Accum: %0.3f  TrainTo: %0.3f",
                i, state.current_player, v0, state.accum_luck, value);
          }
          num_states += 1;
        }

        for (int stage = 0; stage < stage_count; ++stage) {
          // Scale for the length of the game
          int index = (trajectory->states.size() - 1) *
                      static_cast<double>(stage) / (stage_count - 1);
          const Trajectory::State& s = trajectory->states[index];
          value_accuracies[stage].Add(
              (s.value >= 0) == (trajectory->returns[s.current_player] >= 0));
          value_predictions[stage].Add(abs(s.value));
        }
      }
    }
    absl::Time now = absl::Now();
    double seconds = absl::ToDoubleSeconds(now - last);

    logger.Print("Step: %d", step);
    logger.Print(
        "Collected %5d states from %3d games, %.1f states/s; "
        "%.1f states/(s*actor), game length: %.1f",
        num_states, num_trajectories, num_states / seconds,
        num_states / (config.actors * seconds),
        static_cast<double>(num_states) / num_trajectories);
    logger.Print("Queue size: %d. Buffer size: %d. States seen: %d", queue_size,
                 replay_buffer.Size(), replay_buffer.TotalAdded());

    if (stop->StopRequested()) {
      break;
    }

    last = now;

    replay_buffer.SaveBuffer(config.path + "/replay_buffer.data");

    VPNetModel::LossInfo losses;
    {  // Extra scope to return the device for use for inference asap.
      DeviceManager::DeviceLoan learn_model =
          device_manager->Get(config.train_batch_size, device_id);

      // Let the device manager know that the first device is now
      // off-limits for inference and should only be used for learning
      // (if config.explicit_learning == true).
      device_manager->SetLearning(config.explicit_learning);

      // Learn from them.
      for (int i = 0; i < replay_buffer.Size() / config.train_batch_size; i++) {
        losses += learn_model->Learn(
            replay_buffer.Sample(&rng, config.train_batch_size));
      }

      // The device manager can now once again use the first device for
      // inference (if it could not before).
      device_manager->SetLearning(false);
    }

    // Always save a checkpoint, either for keeping or for loading the weights
    // to the other sessions. It only allows numbers, so use -1 as "latest".
    std::string checkpoint_path = device_manager->Get(0, device_id)
        ->SaveCheckpoint(VPNetModel::kMostRecentCheckpointStep);
    if (step % config.checkpoint_freq == 0) {
      device_manager->Get(0, device_id)->SaveCheckpoint(step);
    }
    for (int i = 0; i < device_manager->Count(); ++i) {
      if (i != device_id) {
        device_manager->Get(0, i)->LoadCheckpoint(checkpoint_path);
      }
    }
    for (int i = 0; i < cpu_device_manager->Count(); ++i) {
      cpu_device_manager->Get(0, i)->LoadCheckpoint(checkpoint_path);
    }
    logger.Print("Checkpoint saved: %s", checkpoint_path);

    DataLogger::Record record = {
        {"step", step},
        {"total_states", replay_buffer.TotalAdded()},
        {"states_per_s", num_states / seconds},
        {"states_per_s_actor", num_states / (config.actors * seconds)},
        {"total_trajectories", total_trajectories},
        {"trajectories_per_s", num_trajectories / seconds},
        {"queue_size", queue_size},
        {"game_length", game_lengths.ToJson()},
        {"game_length_hist", game_lengths_hist.ToJson()},
        {"outcomes", outcomes.ToJson()},
        {"value_accuracy",
         json::TransformToArray(value_accuracies,
                                [](auto v) { return v.ToJson(); })},
        {"value_prediction",
         json::TransformToArray(value_predictions,
                                [](auto v) { return v.ToJson(); })},
        {"eval", json::Object({
                     {"count", eval_results->EvalCount()},
                     {"results", json::CastToArray(eval_results->AvgResults())},
                 })},
        {"batch_size", eval->BatchSizeStats().ToJson()},
        {"batch_size_hist", eval->BatchSizeHistogram().ToJson()},
        {"loss", json::Object({
                     {"value", losses.Value()},
                     {"l2reg", losses.L2()},
                     {"sum", losses.Total()},
                 })},
    };
    eval->ResetBatchSizeStats();
    logger.Print("Losses: value: %.4f, l2: %.4f, sum: %.4f",
                 losses.Value(), losses.L2(), losses.Total());

    LRUCacheInfo cache_info = eval->CacheInfo();
    if (cache_info.size > 0) {
      logger.Print(absl::StrFormat(
          "Cache size: %d/%d: %.1f%%, hits: %d, misses: %d, hit rate: %.3f%%",
          cache_info.size, cache_info.max_size, 100.0 * cache_info.Usage(),
          cache_info.hits, cache_info.misses, 100.0 * cache_info.HitRate()));
      eval->ClearCache();
    }
    record.emplace("cache",
                   json::Object({
                       {"size", cache_info.size},
                       {"max_size", cache_info.max_size},
                       {"usage", cache_info.Usage()},
                       {"requests", cache_info.Total()},
                       {"requests_per_s", cache_info.Total() / seconds},
                       {"hits", cache_info.hits},
                       {"misses", cache_info.misses},
                       {"misses_per_s", cache_info.misses / seconds},
                       {"hit_rate", cache_info.HitRate()},
                   }));

    data_logger.Write(record);
    logger.Print("");
  }
}

bool AlphaZero(AlphaZeroConfig config, StopToken* stop, bool resuming) {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(config.game);

  open_spiel::GameType game_type = game->GetType();
  if (game->NumPlayers() != 2)
    open_spiel::SpielFatalError("AlphaZero can only handle 2-player games.");
  if (game_type.reward_model != open_spiel::GameType::RewardModel::kTerminal)
    open_spiel::SpielFatalError("Game must have terminal rewards.");
  if (game_type.dynamics != open_spiel::GameType::Dynamics::kSequential)
    open_spiel::SpielFatalError("Game must have sequential turns.");

  file::Mkdirs(config.path);
  if (!file::IsDirectory(config.path)) {
    std::cerr << config.path << " is not a directory." << std::endl;
    return false;
  }

  std::cout << "Logging directory: " << config.path << std::endl;

  // TODO(mattrek): File vpnet.pb doesn't get updated when config.json changes.
  // => BUG: Changing learning_rate, weight_decay in config.json has no effect.
  if (config.graph_def.empty()) {
    config.graph_def = "vpnet.pb";
    std::string model_path = absl::StrCat(config.path, "/", config.graph_def);
    if (file::Exists(model_path)) {
      std::cout << "Overwriting existing model: " << model_path << std::endl;
    } else {
      std::cout << "Creating model: " << model_path << std::endl;
    }
    SPIEL_CHECK_TRUE(CreateGraphDef(
        *game, config.learning_rate, config.weight_decay, config.path,
        config.graph_def, config.nn_model, config.nn_width, config.nn_depth));
  } else {
    std::string model_path = absl::StrCat(config.path, "/", config.graph_def);
    if (file::Exists(model_path)) {
      std::cout << "Using existing model: " << model_path << std::endl;
    } else {
      std::cout << "Model not found: " << model_path << std::endl;
    }
  }

  std::cout << "Playing game: " << config.game << std::endl;

  config.inference_batch_size = std::max(
      1,
      std::min(config.inference_batch_size, config.actors + config.evaluators));

  config.inference_threads =
      std::max(1, std::min(config.inference_threads,
                           (1 + config.actors + config.evaluators) / 2));

  {
    file::File fd(config.path + "/config.json", "w");
    fd.Write(json::ToString(config.ToJson(), true) + "\n");
  }

  StartInfo start_info = {/*start_time=*/absl::Now(),
                          /*start_step=*/1,
                          /*model_checkpoint_step=*/0,
                          /*total_trajectories=*/0};
  if (resuming) {
    start_info = StartInfoFromLearnerJson(config.path);
  }

  DeviceManager device_manager;
  for (const absl::string_view& device : absl::StrSplit(config.devices, ',')) {
    device_manager.AddDevice(
        VPNetModel(*game, config.path, config.graph_def, std::string(device)));
  }

  if (device_manager.Count() == 0) {
    std::cerr << "No devices specified?" << std::endl;
    return false;
  }

  // The explicit_learning option should only be used when multiple
  // devices are available (so that inference can continue while
  // also undergoing learning).
  if (device_manager.Count() <= 1 && config.explicit_learning) {
    std::cerr << "Explicit learning can only be used with multiple devices."
              << std::endl;
    return false;
  }

  DeviceManager cpu_device_manager;
  cpu_device_manager.AddDevice(
        VPNetModel(*game, config.path, config.graph_def, "/cpu:0"));

  std::cerr << "Loading model from step " << start_info.model_checkpoint_step
            << std::endl;
  {  // Make sure they're all in sync.
    if (!resuming) {
      device_manager.Get(0)->SaveCheckpoint(start_info.model_checkpoint_step);
    }
    for (int i = 0; i < device_manager.Count(); ++i) {
      device_manager.Get(0, i)->LoadCheckpoint(
          start_info.model_checkpoint_step);
    }
    for (int i = 0; i < cpu_device_manager.Count(); ++i) {
      cpu_device_manager.Get(0, i)->LoadCheckpoint(
          start_info.model_checkpoint_step);
    }
  }

  auto eval = std::make_shared<VPNetEvaluator>(
      &device_manager, config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  // mattrek: Unbatched inference always slower on gpu; use cpu for actors,evaluators.
  bool useCpuOnlyInference = config.inference_batch_size <= 1
      && device_manager.Get(0,0)->Device().find("cpu") == std::string::npos;
  if (useCpuOnlyInference) {
    std::cerr << "Using cpu_only inference for actors/evaluators." << std::endl;
  }

  auto inf_eval = std::make_shared<VPNetEvaluator>(
      useCpuOnlyInference ? &cpu_device_manager : &device_manager,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);
  /*
  std::vector<std::shared_ptr<VPNetEvaluator>> inf_evals;
  inf_evals.reserve(config.actors + config.evaluators);
  for (int i = 0; i < config.actors + config.evaluators; ++i) {
    inf_evals.push_back(std::make_shared<VPNetEvaluator>(
        useCpuOnlyInference ? &cpu_device_manager : &device_manager,
        config.inference_batch_size, config.inference_threads,
        config.inference_cache, (config.actors + config.evaluators) / 16));
  }
  */

  ThreadedQueue<Trajectory> trajectory_queue(config.replay_buffer_size /
                                             config.replay_buffer_reuse);

  EvalResults eval_results(config.eval_levels, config.evaluation_window);

  std::vector<Thread> actors;
  actors.reserve(config.actors);
  for (int i = 0; i < config.actors; ++i) {
    actors.emplace_back(
        [&, i]() { actor(*game, config, i, &trajectory_queue, inf_eval, stop); });
  }
  std::vector<Thread> evaluators;
  evaluators.reserve(config.evaluators);
  for (int i = 0; i < config.evaluators; ++i) {
    evaluators.emplace_back(
        [&, i]() { evaluator(*game, config, i, &eval_results, inf_eval, stop); });
  }
  learner(*game, config, &device_manager, &cpu_device_manager, eval, &trajectory_queue,
          &eval_results, stop, start_info, /*verbose=*/false);

  if (!stop->StopRequested()) {
    stop->Stop();
  }

  // Empty the queue so that the actors can exit.
  trajectory_queue.BlockNewValues();
  trajectory_queue.Clear();

  std::cout << "Joining all the threads." << std::endl;
  for (auto& t : actors) {
    t.join();
  }
  for (auto& t : evaluators) {
    t.join();
  }
  std::cout << "Exiting cleanly." << std::endl;
  return true;
}

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel
