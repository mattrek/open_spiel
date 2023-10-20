set(GAME_SOURCES
  backgammon.cc
  backgammon.h
  tic_tac_toe.cc
  tic_tac_toe.h
)

add_library (games OBJECT ${GAME_SOURCES})

target_include_directories (games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(backgammon_test backgammon_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(backgammon_test backgammon_test)
