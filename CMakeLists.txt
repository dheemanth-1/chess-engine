cmake_minimum_required(VERSION 3.10)
project(ChessEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Just tell the compiler where to find chess.hpp
include_directories(extern/chess-library-master/include)

add_executable(chess-engine 
    src/main.cpp
    src/evaluation.cpp
    src/search.cpp
    src/eval_dispatch.cpp
    src/nnue.cpp
)

if(MSVC)
    # 1. FIX: Enable exception handling to clear Warning C4530
    target_compile_options(chess-engine PRIVATE /EHsc)

    target_compile_options(chess-engine PRIVATE 
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/arch:AVX2>
        $<$<CONFIG:Release>:/fp:fast>
    )
else()
    # 2. FIX: Changed Windows '/' flags to Unix '-' flags for GCC/Clang compatibility
    target_compile_options(chess-engine PRIVATE 
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-mavx2>
        $<$<CONFIG:Release>:-mfma>
        $<$<CONFIG:Release>:-ffast-math>
    )
endif()

# ── Compiler optimizations ────────────────────────────────────────────────────
# AVX2 lets the CPU process 8 floats per instruction instead of 1.
# FMA (Fused Multiply-Add) combines multiply and add in a single instruction.
# Both are supported by the i5-11320H (Tiger Lake) and all recent Intel/AMD CPUs.
# fast-math allows the compiler to reorder float operations for speed
# (safe for our use case — we don't care about strict IEEE754 ordering).
# if(MSVC)
#     target_compile_options(chess-engine PRIVATE
#         /O2          # Full optimization
#         /arch:AVX2   # Enable AVX2 + FMA instructions
#         /fp:fast     # Aggressive float optimization
#     )
# else()
#     target_compile_options(chess-engine PRIVATE
#         -O3          # Full optimization
#         -mavx2       # Enable AVX2 instructions
#         -mfma        # Enable FMA instructions
#         -ffast-math  # Aggressive float optimization
#     )
# endif()

