cmake_minimum_required(VERSION 3.14)
project(FaceReco)

set(CMAKE_CXX_STANDARD 17)

# Enable Qt's automatic processing features
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Set path for ONNX Runtime
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")

# Required packages
find_package(Qt5 COMPONENTS Core Gui Widgets PrintSupport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(CURL REQUIRED libcurl)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ui
    ${ONNXRUNTIME_DIR}/include
    third_party/onnxruntime/include
    /path/to/onnxruntime/include  # ← Replace this with actual path if needed
)

# Link directories
link_directories(
    ${ONNXRUNTIME_DIR}/lib
    /path/to/onnxruntime/lib       # ← Replace this with actual path if needed
)

# Executable and source files
add_executable(FaceReco
    main.cpp
    ui/mainwindow.cpp
    ui/mainwindow.hpp
    src/crawler.cpp
    src/face_embedder.cpp
    src/crawler_worker.cpp
    include/crawler_worker.hpp    # Ensures Q_OBJECT gets moc-processed
)

# Link libraries
target_link_libraries(FaceReco
    PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    Qt5::PrintSupport
    ${OpenCV_LIBS}
    onnxruntime
    ${CURL_LIBRARIES}
)

# Compiler settings
target_include_directories(FaceReco PRIVATE 
    ${Qt5Widgets_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

target_compile_definitions(FaceReco PRIVATE ${Qt5Widgets_DEFINITIONS})
target_compile_options(FaceReco PRIVATE ${CURL_CFLAGS_OTHER})
