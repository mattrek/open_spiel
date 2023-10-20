add_library (game_transforms OBJECT
  coop_to_1p.cc
  coop_to_1p.h
  efg_writer.cc
  efg_writer.h
  misere.cc
  misere.h
  normal_form_extensive_game.cc
  normal_form_extensive_game.h
  repeated_game.cc
  repeated_game.h
  restricted_nash_response.cc
  restricted_nash_response.h
  start_at.cc
  start_at.h  
  turn_based_simultaneous_game.cc
  turn_based_simultaneous_game.h
)
target_include_directories (game_transforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

