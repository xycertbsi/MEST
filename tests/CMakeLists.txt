add_executable(test_mest test_mest.cu)
target_link_libraries(test_mest PRIVATE mest)
target_include_directories(test_mest PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_mest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

enable_testing()
add_test(NAME MestTests COMMAND test_mest)
