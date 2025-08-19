cmake_minimum_required(VERSION 3.5)

project(P4-xj)

SET ( TARGET P4-xj )
# SET ( EXECUTABLE_OUTPUT_PATH /home/easyits/ai/wpf/Employee/road4-master/integrate/release)
SET ( CROSS_TOOL_PATH_PREFIX /home/easyits/ai/samba/novaic/arm-ca9-linux-gnueabihf-6.5/bin )
SET ( CMAKE_C_COMPILER   ${CROSS_TOOL_PATH_PREFIX}/arm-ca9-linux-gnueabihf-gcc )
SET ( CMAKE_CXX_COMPILER ${CROSS_TOOL_PATH_PREFIX}/arm-ca9-linux-gnueabihf-g++ )
set(CMAKE_C_FLAGS "-mcpu=cortex-a9 -fno-aggressive-loop-optimizations -W -Wall -O3 -D__LINUX -std=c++11 -finline-functions -fPIC -Wno-aggressive-loop-optimizations -D__LINUX_USER__ -mfpu=neon-fp16 -fpermissive -Wsign-compare -Wno-missing-braces -Wstrict-prototypes -ffunction-sections -fdata-sections -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "-mcpu=cortex-a9 -fno-aggressive-loop-optimizations -W -Wall -O3 -D__LINUX -std=c++11 -finline-functions -fPIC -Wno-aggressive-loop-optimizations -D__LINUX_USER__ -mfpu=neon-fp16 -fpermissive -Wsign-compare -Wno-missing-braces -Wstrict-prototypes -ffunction-sections -fdata-sections -fvisibility=hidden")
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE DEBUG)

# cv sys api .h .so
set(RMAI_API_PATH   /home/easyits/ai/samba/third_party/rmai_lib_novatek_9852X)
include_directories(${RMAI_API_PATH}/include include
    /home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/common
    /home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/deduplication
    /home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/detector
    /home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/interface
    /home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/model
    /home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/object_tracker
    /home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/object_tracker/iou_tracker
    /home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/object_tracker/iou_tracker/sort_tracker)
set(CV_COMMON_LIB   ${RMAI_API_PATH}/lib/libcv_sys_api.so)      # rmImageCreate
set(RMAI_NT_LIB     ${RMAI_API_PATH}/lib/librmai_novt.so)       # novt_hwcnn::CNN::Run

aux_source_directory(src SOURCES)
aux_source_directory(/home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/common  SOURCES)
aux_source_directory(/home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/deduplication SOURCES)
aux_source_directory(/home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/detector  SOURCES)
aux_source_directory(/home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/interface SOURCES)
aux_source_directory(/home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/model SOURCES)
aux_source_directory(/home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/object_tracker/iou_tracker SOURCES)
aux_source_directory(/home/easyits/ai/wpf/Employee/01.NOVT/P4-xj/so/src/object_tracker/iou_tracker/sort_tracker SOURCES)
# add_executable(${TARGET} main.cpp ${SOURCES})       
add_library(${TARGET} SHARED ${SOURCES})      

# OpenCV
set( OpenCV_VERSION_MAJOR     2    )
set( OpenCV_VERSION_MINOR     4    )
set( OpenCV_VERSION_REVISION  11   )
set( OpenCV_INCLUDE_DIRS      /home/easyits/ai/samba/third_party/opencv2411/include )
set( OpenCV_INSTALL_PATH      /home/easyits/ai/samba/third_party/opencv2411/lib_novatek )
set( OpenCV_VERSION "${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.${OpenCV_VERSION_REVISION}")
set( OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} /home/easyits/ai/samba/third_party/opencv2411/include/opencv )
set( OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} /home/easyits/ai/samba/third_party/opencv2411/include/opencv2 )
set( OpenCV_LIBS                 ${OpenCV_INSTALL_PATH}/libopencv_objdetect.a  )
set( OpenCV_LIBS  ${OpenCV_LIBS} ${OpenCV_INSTALL_PATH}/libopencv_highgui.a    )
set( OpenCV_LIBS  ${OpenCV_LIBS} ${OpenCV_INSTALL_PATH}/libopencv_video.a      )
set( OpenCV_LIBS  ${OpenCV_LIBS} ${OpenCV_INSTALL_PATH}/libopencv_imgproc.a    )
set( OpenCV_LIBS  ${OpenCV_LIBS} ${OpenCV_INSTALL_PATH}/libopencv_core.a       )
set( OpenCV_LIBS  ${OpenCV_LIBS} ${OpenCV_INSTALL_PATH}/libzlib.a              )
set( OpenCV_LIBS  ${OpenCV_LIBS} ${OpenCV_INSTALL_PATH}/liblibjpeg.a           )
set( OpenCV_LIBS  ${OpenCV_LIBS} ${OpenCV_INSTALL_PATH}/liblibpng.a            )
set( OpenCV_LIBS  ${OpenCV_LIBS} ${OpenCV_INSTALL_PATH}/liblibjasper.a         )
set( OpenCV_LIBS  ${OpenCV_LIBS} ${OpenCV_INSTALL_PATH}/libopencv_features2d.a )
set( OpenCV_FOUND TRUE)
set( CMAKE_EXE_LINKER_FLAGS "-lpthread")
message(STATUS "OpenCV  library: ${OpenCV_INSTALL_PATH}")
message(STATUS "        version: ${OpenCV_VERSION}")
message(STATUS "        libraries: ${OpenCV_LIBS}")
message(STATUS "        include path: ${OpenCV_INCLUDE_DIRS}")

if(OpenCV_FOUND)
    target_include_directories(${TARGET} PRIVATE ${OpenCV_INCLUDE_DIRS})
#    target_link_libraries(${TARGET} "-Wl,--allow-shlib-undefined"
#                                    ${OpenCV_LIBS} 
#                                    ${RMAIOP_CNN_LIB}
#                                    ${CV_COMMON_LIB}
#                                    ${RMAI_NT_LIB} )
    target_link_libraries(${TARGET} 
                                    "-Wl,--allow-shlib-undefined"
                                    # "-Wl,--no-undefined"
                                    ${OpenCV_LIBS} 
                                    ${CV_COMMON_LIB}
                                    ${RMAI_NT_LIB} )
else()
    message(WARNING "OpenCV not found, examples won't be built")
endif()
