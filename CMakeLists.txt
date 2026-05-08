cmake_minimum_required(VERSION 3.18)
project(MEST VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MEST_BUILD_PYTHON  "Build Python bindings"  ON)
option(MEST_BUILD_JAVA    "Build Java bindings"    ON)
option(MEST_BUILD_EXAMPLES "Build examples"        ON)
option(MEST_BUILD_TESTS   "Build tests"            ON)

find_package(CUDAToolkit REQUIRED)

set(MEST_SOURCES
    src/kernels.cu
    src/capi.cu
)

add_library(mest SHARED ${MEST_SOURCES})

target_include_directories(mest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(mest PUBLIC CUDA::cudart)

set_target_properties(mest PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "mest"
)

if(MEST_BUILD_JAVA)
    find_package(Java QUIET)
    find_package(JNI QUIET)
    if(JNI_FOUND)
        add_library(mest_jni SHARED src/jni_bridge.cu)
        target_include_directories(mest_jni PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${JNI_INCLUDE_DIRS}
        )
        target_link_libraries(mest_jni PRIVATE mest CUDA::cudart)
        set_target_properties(mest_jni PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            OUTPUT_NAME "mest_jni"
        )
    else()
        message(STATUS "JNI not found — skipping Java bindings")
    endif()
endif()

if(MEST_BUILD_EXAMPLES)
    add_subdirectory(examples/cpp)
endif()

if(MEST_BUILD_TESTS)
    add_subdirectory(tests)
endif()

install(TARGETS mest
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/mest DESTINATION include)
