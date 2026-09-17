cmake_minimum_required(VERSION 3.18)
project(TopologicalComputation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# ═══════════════════════════════════════════════════════════════════════════
# Layer architecture (all header-only INTERFACE libraries)
#
#   L0: topcomp_core     — complex, constants, mir_group, phase_space, cocycle,
#                           hilbert, measurement, gates, prng, machine, witt,
#                           weyl, category, lie_algebra, representation, compiler
#   L1: topcomp_contract — state_contract, descriptor, emission, structure,
#                           sector, fiber
#   L2: topcomp_math     — exact_decomposition, kernel_calculus
#   L3: topcomp_doctrine — semantic_kernel
#   L4: topcomp_ir       — platform_ir, semantic_passes
#   L5: topcomp_runtime  — runtime, machine_state, hardware_topology,
#                           fibered_memory, regime_planner, spectral_transport,
#                           persistent_format, pipeline
#   L6: topcomp_api      — topcomp_api, c_api, exemplars
#
# Currently all resolve to the same include tree.  When the project
# matures these become separate static/shared libraries.
# ═══════════════════════════════════════════════════════════════════════════

add_library(topcomp INTERFACE)
target_include_directories(topcomp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Logical layer targets (alias into the single header tree for now)
foreach(_layer topcomp_core topcomp_contract topcomp_math
               topcomp_doctrine topcomp_ir topcomp_runtime topcomp_api)
    add_library(${_layer} INTERFACE)
    target_link_libraries(${_layer} INTERFACE topcomp)
endforeach()

# Main executable
add_executable(mir_compute
    src/kernels.cu
    src/main.cu
)
target_link_libraries(mir_compute PRIVATE topcomp CUDA::cudart CUDA::curand)
target_compile_options(mir_compute PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
)

# Optional: tests
option(BUILD_TESTS "Build test suite" ON)
if(BUILD_TESTS)
    enable_testing()
    add_executable(mir_tests
        src/kernels.cu
        src/tests.cu
    )
    target_link_libraries(mir_tests PRIVATE topcomp CUDA::cudart CUDA::curand)
    target_compile_options(mir_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
    )
    add_test(NAME mir_tests COMMAND mir_tests)
endif()
