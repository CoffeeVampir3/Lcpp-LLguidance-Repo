cmake_minimum_required(VERSION 3.14.0)
project(working_example)
set(CMAKE_CXX_STANDARD 17)

set(LLAMACPP_REPO "https://github.com/ggerganov/llama.cpp.git" CACHE STRING "llama.cpp repository URL")
message(STATUS "Using llama.cpp repo ${LLAMACPP_REPO}")

set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama.cpp: build examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama.cpp: build tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama.cpp: build server" FORCE)

# Enable common
set(LLAMA_BUILD_COMMON ON CACHE BOOL "llama.cpp: build common utils library" FORCE)

# OPTIONAL DEP REQUIRES RUST
set(LLAMA_LLGUIDANCE ON CACHE BOOL "llama.cpp: enable LLGuidance support" FORCE)
find_program(CARGO cargo)
if(NOT CARGO)
    message(WARNING "LLGuidance needs rust. GO GET RUST. WTF R U DOING https://rustup.rs/")
    set(LLAMA_LLGUIDANCE OFF CACHE BOOL "llama.cpp: enable LLGuidance support" FORCE)
endif()

include(FetchContent)
FetchContent_Declare(
        llama
        GIT_REPOSITORY ${LLAMACPP_REPO}
)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

FetchContent_MakeAvailable(llama)

add_executable(working_example
        working_example.cpp
)
target_link_libraries(working_example PUBLIC llama common)
target_include_directories(working_example PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${llama_SOURCE_DIR}
)

add_executable(broken_example
        broken_example.cpp
)
target_link_libraries(broken_example PUBLIC llama common)
target_include_directories(broken_example PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${llama_SOURCE_DIR}
)

add_executable(fixed_example
        fixed_example.cpp
)
target_link_libraries(fixed_example PUBLIC llama common)
target_include_directories(fixed_example PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${llama_SOURCE_DIR}
)