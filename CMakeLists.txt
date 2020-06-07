macro(print_all_with prefix)
    message(STATUS "print variables with ${prefix} ------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        string(FIND "${_variableName}" "${prefix}" matched_string_pos )
        if("${matched_string_pos}" GREATER "-1")
                message(STATUS "${_variableName}=${${_variableName}}")
        endif()
    endforeach()
    message(STATUS "--------------------------------------------------------------}")
endmacro()

macro(init_cmake_submodule)
    message(STATUS "Checking if cmake submodule is present...")
    file(GLOB CMAKE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*")
    list(LENGTH CMAKE_FILES N_CMAKE_FILES)
    if(N_CMAKE_FILES EQUAL 0)
        message(STATUS "Git submodule initialization")
        execute_process(COMMAND git submodule init)
        execute_process(COMMAND git submodule update)
    endif()
endmacro()
###########################################################
## Main Program
# Required packages
# cmake >= 3.1, g++ >= 7, ninja >= 1.10
# torch 1.5
###########################################################
cmake_minimum_required( VERSION 3.1.0 )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

###########################################################
## CLANG/CLANG++
#set(CMAKE_C_COMPILER "/usr/bin/clang-6.0")
#set(CMAKE_CXX_COMPILER "/usr/bin/clang++-6.0")
###########################################################
project(pytorch_cpp VERSION 0.9.0)
if (MSVC)
        set(CMAKE_PREFIX_PATH "WINDOWS/ABSOLUTE/PATH/TO/libtorch")
else (MSVC)
        set(CMAKE_PREFIX_PATH "/home/james/Coding/libtorch")
endif (MSVC)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Torch REQUIRED)
# Compiler Flags
if (MSVC)
        set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Wall /MP ${TORCH_CXX_FLAGS}" )
        set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING")
        set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /MP ${TORCH_CXX_FLAGS}" )
else (MSVC)
        # -g - Debug information for gdb
        set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -g -pedantic" )
        set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${TORCH_CXX_FLAGS}" )
        set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" )
        set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${TORCH_CXX_FLAGS}" )
endif (MSVC)
# Include source, header, and test source files
include_directories(
                    "${PROJECT_SOURCE_DIR}/inc"
                    "${PROJECT_SOURCE_DIR}/src" 
                    )
file(GLOB ALL_SRCS
        "${PROJECT_SOURCE_DIR}/inc/*.h"
        "${PROJECT_SOURCE_DIR}/inc/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.c"
        )
file(GLOB ALL_TESTS
        "${PROJECT_SOURCE_DIR}/test/*.cpp"
        )

add_executable( "${PROJECT_NAME}" ${ALL_SRCS}) ## make sure to include all src files
target_link_libraries("${PROJECT_NAME}" ${TORCH_LIBRARIES})
## Target Properties and Linking
set_property(TARGET "${PROJECT_NAME}" PROPERTY CXX_STANDARD 14)
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)
set( CMAKE_VERBOSE_MAKEFILE ON ) # we want to see compiler flags

###########################################################
## Unit Tests
###########################################################
init_cmake_submodule()
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# Remove main from src
list(REMOVE_ITEM ALL_SRCS "${PROJECT_SOURCE_DIR}/src/main.cpp")
enable_testing()
include(AddGoogleTest)
add_executable("${PROJECT_NAME}_test" ${ALL_TESTS} ${ALL_SRCS})
target_link_libraries("${PROJECT_NAME}_test" ${TORCH_LIBRARIES} gtest gtest_main gmock)
add_test(NAME "${PROJECT_NAME}_test" COMMAND "${PROJECT_NAME}_test")
