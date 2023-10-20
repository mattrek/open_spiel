# To enable C++ Torch AlphaZero, you will need to set OPEN_SPIEL_BUILD_WITH_LIBTORCH.
if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  if(NOT OPEN_SPIEL_BUILD_WITH_LIBNOP)
    message(FATAL_ERROR
      "ziggy_torch requires libnop (OPEN_SPIEL_BUILD_WITH_LIBNOP)")
  endif()

  add_library (ziggy_torch OBJECT
    ziggy.h
    ziggy.cc
    device_manager.h
    model.h
    model.cc
    v_evaluator.h
    v_evaluator.cc
    v_net.h
    v_net.cc
  )
  target_include_directories (ziggy_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(torch_ziggy_test ziggy_test.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:ziggy_torch> $<TARGET_OBJECTS:tests>)
  add_test(torch_ziggy_test torch_ziggy_test)

  add_executable(torch_ziggy_model_test model_test.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:ziggy_torch> $<TARGET_OBJECTS:tests>)
  add_test(torch_ziggy_model_test torch_ziggy_model_test)

  add_executable(torch_ziggy_v_net_test v_net_test.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:ziggy_torch> $<TARGET_OBJECTS:tests>)
  add_test(torch_ziggy_v_net_test torch_ziggy_v_net_test)

  target_link_libraries (ziggy_torch ${TORCH_LIBRARIES})
  target_link_libraries (torch_ziggy_test ${TORCH_LIBRARIES})
  target_link_libraries (torch_ziggy_model_test ${TORCH_LIBRARIES})
  target_link_libraries (torch_ziggy_v_net_test ${TORCH_LIBRARIES})  
endif ()
