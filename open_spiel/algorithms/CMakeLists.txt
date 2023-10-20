add_library (algorithms OBJECT
  best_response.cc
  best_response.h
  cfr.cc
  cfr.h
  cfr_br.cc
  cfr_br.h
  corr_dist.cc
  corr_dist.h
  corr_dist/afcce.cc
  corr_dist/afcce.h
  corr_dist/afce.cc
  corr_dist/afce.h
  corr_dist/efcce.cc
  corr_dist/efcce.h
  corr_dist/efce.cc
  corr_dist/efce.h
  corr_dist/cce.cc
  corr_dist/cce.h
  corr_dist/ce.cc
  corr_dist/ce.h
  corr_dev_builder.cc
  corr_dev_builder.h
  deterministic_policy.cc
  deterministic_policy.h
  evaluate_bots.cc
  evaluate_bots.h
  expected_returns.cc
  expected_returns.h
  external_sampling_mccfr.cc
  external_sampling_mccfr.h
  fsicfr.cc
  fsicfr.h
  get_all_histories.cc
  get_all_histories.h
  get_all_infostates.cc
  get_all_infostates.h
  get_all_states.cc
  get_all_states.h
  get_legal_actions_map.cc
  get_legal_actions_map.h
  history_tree.cc
  history_tree.h
  infostate_tree.h
  infostate_tree.cc
  is_mcts.cc
  is_mcts.h
  matrix_game_utils.cc
  matrix_game_utils.h
  nfg_writer.cc
  nfg_writer.h
  mcts.cc
  mcts.h
  minimax.cc
  minimax.h
  observation_history.h
  observation_history.cc
  oos.h
  oos.cc
  outcome_sampling_mccfr.cc
  outcome_sampling_mccfr.h
  policy_iteration.cc
  policy_iteration.h
  state_distribution.cc
  state_distribution.h
  tabular_best_response_mdp.cc
  tabular_best_response_mdp.h
  tabular_exploitability.cc
  tabular_exploitability.h
  tabular_q_learning.cc
  tabular_q_learning.h
  tabular_sarsa.cc
  tabular_sarsa.h
  tensor_game_utils.cc
  tensor_game_utils.h
  trajectories.cc
  trajectories.h
  value_iteration.cc
  value_iteration.h
)
target_include_directories (algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (${OPEN_SPIEL_BUILD_WITH_ORTOOLS})
  add_subdirectory (ortools)
endif()

add_executable(mcts_test mcts_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(mcts_test mcts_test)

add_subdirectory (alpha_zero)
add_subdirectory (alpha_zero_torch)
add_subdirectory (dqn_torch)
add_subdirectory (ziggy_torch)
