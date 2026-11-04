cmake_minimum_required(VERSION 3.18)

# Va antes de project(): con MSVC, project() fija Debug si no hay nada definido.
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de compilacion")
endif()

project(FiltrosImagen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(filtros STATIC filtros.cpp filtros.cu)

add_executable(main main.cpp)
add_executable(pruebas pruebas.cpp)
add_executable(benchmark benchmark.cpp)

foreach(programa main pruebas benchmark)
    target_link_libraries(${programa} PRIVATE filtros)
    set_target_properties(${programa} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
    )
endforeach()
