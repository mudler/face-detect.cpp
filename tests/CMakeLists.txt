function(fd_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE facedetect)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

fd_add_test(test_smoke)
fd_add_test(test_nms)
fd_add_test(test_align)
fd_add_test(test_preprocess)
fd_add_test(test_scrfd_graph)
fd_add_test(test_arcface_stem)
fd_add_test(test_arcface_block)
fd_add_test(test_embedding)
fd_add_test(test_sface_embedding)
fd_add_test(test_genderage)
fd_add_test(test_antispoof)
fd_add_test(test_landmarks)
fd_add_test(test_detect)
fd_add_test(test_decode)
fd_add_test(test_yunet)
fd_add_test(test_cache_safety)

# Model/baseline-dependent tests read fixtures + baselines from the project root
# and carry the "model" label so CI can exclude them with `ctest -LE model`.
set_tests_properties(test_align PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_scrfd_graph PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_arcface_stem PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_arcface_block PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_embedding PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_cache_safety PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_sface_embedding PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_genderage PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_antispoof PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_landmarks PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_detect PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_yunet PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(test_decode PROPERTIES LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Python converter check (skips with exit 77 when the venv/model are absent).
find_program(FACEDETECT_PY NAMES python3 HINTS ${CMAKE_SOURCE_DIR}/.venv/bin)
if(FACEDETECT_PY)
  add_test(NAME check_convert COMMAND ${FACEDETECT_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_convert.py)
  set_tests_properties(check_convert PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
endif()
