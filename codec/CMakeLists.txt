cmake_minimum_required(VERSION 2.6)

project(ppp-video)

add_definitions(-D_POSIX_C_SOURCE=200809L)

# Define pre-processor symbol with path to OpenCL kernels
# in install location.
add_definitions(-DKERNELS_PATH="${CMAKE_INSTALL_PREFIX}/kernels")

set(CMAKE_C_FLAGS "-g -std=c99 -pedantic -Wall -O3 -fomit-frame-pointer")

include_directories("${PROJECT_SOURCE_DIR}/ppp_pnm"
  "${PROJECT_SOURCE_DIR}/image" "${PROJECT_SOURCE_DIR}/video")

# Create a source file if it does not exists so it can be
# installed and we can add custom dependencies for it.
function(create_source filename)
  set(full "${PROJECT_SOURCE_DIR}/${filename}")
  # Create the file if it does not exist
  if(NOT EXISTS "${full}")
    file(APPEND "${full}" "")
  endif(NOT EXISTS "${full}")
endfunction(create_source)

# Libraries to build

add_library(ppp_pnm ppp_pnm/ppp_pnm.c)

add_library(ppp_image
  image/ppp_image.c
  image/frame_encoding.c
  image/compression_stats.c
  image/ocl_init.c
)

add_library(ppp_video
  image/motion_stats.c
  image/ppp_video.c
  image/video_load.c
)

# Executables to build

add_executable(dct_flops image/dct_flops.c)
target_link_libraries(dct_flops ppp_image m)

add_executable(image_encoder image/image_encoder.c)
target_link_libraries(image_encoder OpenCL ppp_pnm ppp_image m)

add_executable(viewer image/viewer.c)
target_link_libraries(viewer
  ppp_pnm ppp_image ppp_video m SDL avformat avcodec avutil swscale)

# Create empty solution files if they do not exist
create_source(solution/image_encoder_kernels.cl)

install(TARGETS dct_flops image_encoder viewer RUNTIME DESTINATION bin)
install(FILES "solution/image_encoder_kernels.cl" DESTINATION kernels)

# Custom dummy target to make IDEs (such as QtCreater)
# recognize the image encoder's OpenCL file as additional source
add_custom_target(DummyI SOURCES solution/image_encoder_kernels.cl)

# Build video_encoder only when video/ directory is present
if(EXISTS "${PROJECT_SOURCE_DIR}/video")
  # Create empty solution files if they do not exist
  create_source(solution/video_encoder_cl.c)
  create_source(solution/video_encoder_kernels.cl)

  add_executable(video_encoder
    video/video_encoder.c
    solution/video_encoder_cl.c
  )
  target_link_libraries(video_encoder
    OpenCL m ppp_pnm ppp_image ppp_video avformat avcodec avutil swscale)

  install(TARGETS video_encoder RUNTIME DESTINATION bin)
  install(FILES solution/video_encoder_kernels.cl DESTINATION kernels)

  # Make IDEs recognize the video encoder's OpenCL file
  add_custom_target(DummyV SOURCES solution/video_encoder_kernels.cl)
endif(EXISTS "${PROJECT_SOURCE_DIR}/video")


