if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  add_executable(alpha_zero_torch_example alpha_zero_torch_example.cc
                ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch>)
  target_link_libraries (alpha_zero_torch_example ${TORCH_LIBRARIES})

  add_executable(alpha_zero_torch_game_example alpha_zero_torch_game_example.cc
                ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch>
                $<TARGET_OBJECTS:bots>)
  target_link_libraries (alpha_zero_torch_game_example ${TORCH_LIBRARIES})

  add_executable(dqn_torch_example dqn_torch_example.cc
                ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:dqn_torch>)
  target_link_libraries (dqn_torch_example ${TORCH_LIBRARIES})

  add_executable(ziggy_torch_example ziggy_torch_example.cc
                ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:ziggy_torch>)
  target_link_libraries (ziggy_torch_example ${TORCH_LIBRARIES})

  add_executable(ziggy_torch_game_example ziggy_torch_game_example.cc
                ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:ziggy_torch>
                $<TARGET_OBJECTS:bots>)
  target_link_libraries (ziggy_torch_game_example ${TORCH_LIBRARIES})
endif ()
