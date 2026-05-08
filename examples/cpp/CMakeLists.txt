add_executable(mest_hello hello.cu)
target_link_libraries(mest_hello PRIVATE mest)
target_include_directories(mest_hello PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(mest_hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(mest_vector_ops vector_ops.cu)
target_link_libraries(mest_vector_ops PRIVATE mest)
target_include_directories(mest_vector_ops PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(mest_vector_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
