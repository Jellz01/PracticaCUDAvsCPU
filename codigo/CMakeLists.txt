cmake_minimum_required(VERSION 3.18)
project(FiltrosImagen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly set CUDA Toolkit path if needed
#set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4")
#find_package(CUDA REQUIRED)

# Explicitly specify the CUDA host compiler (adjust this path for your system)
set(CUDA_HOST_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe")

add_executable(main main.cpp filtros.cpp )

set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
)
