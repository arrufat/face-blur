cmake_minimum_required(VERSION 3.11)
project("Face Blur" LANGUAGES CXX)

set(CPACK_PACKAGE_NAME "faceblur")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})

message(${PROJECT_NAME} " - v" ${VERSION})

# Use C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." ON)
if(${FORCE_COLORED_OUTPUT})
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
		add_compile_options (-fdiagnostics-color=always)
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		add_compile_options (-fcolor-diagnostics)
	endif()
endif()

# Enable ccache if it exists
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

# OpenMP support
find_package(OpenMP)

include(FetchContent)
# Fetch Dlib
set(DLIB_TAG v19.19)
FetchContent_Declare(
	dlib
	GIT_REPOSITORY https://github.com/davisking/dlib.git
	GIT_TAG        ${DLIB_TAG}
	)
message("-- Fetching Dlib ${DLIB_TAG}")
FetchContent_MakeAvailable(dlib)

include_directories(./src)

# Build the main target
add_executable(faceblur src/faceblur.cpp)
target_link_libraries(faceblur dlib::dlib OpenMP::OpenMP_CXX)
target_compile_options(faceblur PRIVATE -Wall -Wextra -pedantic -Wno-deprecated-copy)
install(TARGETS faceblur DESTINATION bin)

# Package
set(CPACK_PACKAGE_VENDOR "OMNIOUS")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A small tool to process the output of dbtool")
set(CPACK_COMPONENTS_ALL applications libraries headers)
include(CPack)
