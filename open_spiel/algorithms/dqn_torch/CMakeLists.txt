# To enable C++ Torch DQN, you will need to set OPEN_SPIEL_BUILD_WITH_LIBTORCH. 
if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  add_library (dqn_torch OBJECT
    dqn.h
    dqn.cc
    simple_nets.h
    simple_nets.cc
  )
  target_include_directories (dqn_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(dqn_torch ${TORCH_LIBRARIES})
endif()
