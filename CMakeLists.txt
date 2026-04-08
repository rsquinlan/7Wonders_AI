cmake_minimum_required(VERSION 3.5)

project(7Wonders LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include_directories(include)

set(SOURCES
    main.cpp
	node.cpp
	mcts.cpp
	utils.cpp
	src/game.cpp
    src/card.cpp
	src/player.cpp
    src/wonder.cpp
	src/filer.cpp
	)

set(HEADERS
	utils.h
	mcts.h
	node.h
	nn_player.h
	include/game.h
    include/card.h
	include/player.h
	include/resources.h
    include/wonder.h
	include/filer.h
	include/nlohmann/json.hpp
	)

# ── ONNX Runtime (optional — required for NNPlayer / getBestMoveNN) ──────────
# Install: https://github.com/microsoft/onnxruntime/releases
# Set ONNXRUNTIME_ROOT to your extracted directory, e.g.:
#   cmake .. -DONNXRUNTIME_ROOT=/usr/local
set(ONNXRUNTIME_ROOT "/usr/local" CACHE PATH "Root directory of the ONNX Runtime installation")

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
)
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include/onnxruntime"
          "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH
)

if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE_DIR)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
    list(APPEND SOURCES nn_player.cpp)
    add_executable(7Wonders ${SOURCES} ${HEADERS})
    target_include_directories(7Wonders PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(7Wonders PRIVATE "${ONNXRUNTIME_LIB}")
    target_compile_definitions(7Wonders PRIVATE WITH_NN_PLAYER)
else()
    message(STATUS "ONNX Runtime not found — NNPlayer disabled. Set -DONNXRUNTIME_ROOT=<path> to enable.")
    add_executable(7Wonders ${SOURCES} ${HEADERS})
endif()

file(MAKE_DIRECTORY ./build/io ./io ./logs)
