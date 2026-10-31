cmake_minimum_required(VERSION 3.20)
project(gomoku_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/utf-8 /W3 /wd4996 /wd4267 /MP)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/std:c++20>)
else()
    add_compile_options(-Wall -Wextra -O2)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -DGK_ASAN=ON 时启用 AddressSanitizer（只给 gk 与 gk_core，便于定位越界/悬垂指针）
option(GK_ASAN "build with AddressSanitizer" OFF)
if(GK_ASAN AND MSVC)
    add_compile_options(/fsanitize=address /Zi /Od)
    add_link_options(/fsanitize=address)
    message(STATUS "AddressSanitizer: ON")
endif()

find_package(Torch REQUIRED)
message(STATUS "Torch: ${TORCH_VERSION}  libs=${TORCH_LIBRARIES}")

set(GK_SRC
    src/board.cpp
    src/heuristics.cpp
    src/mcts.cpp
    src/vcf.cpp
    src/dataset.cpp
    src/net_torch.cpp
    src/selfplay.cpp
    src/trainer.cpp
    src/gate.cpp
    src/elo.cpp
    src/elo_store.cpp
    src/elo_run.cpp
    src/batcher.cpp
    src/pipeline.cpp
    src/sysutil.cpp
)

add_executable(gk src/main.cpp)
target_link_libraries(gk PRIVATE gk_core)

add_library(gk_core STATIC ${GK_SRC})
target_include_directories(gk_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(gk_core PUBLIC ${TORCH_LIBRARIES})
target_compile_options(gk_core PUBLIC ${TORCH_CXX_FLAGS})

foreach(t test_rules test_mcts test_vcf test_dataset test_net test_selfplay test_train test_gate probe_mcts probe_eval probe_multi probe_train distill)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE gk_core)
endforeach()
