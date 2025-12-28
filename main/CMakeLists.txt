# Main component CMakeLists.txt for Fan Anomaly Detection

# Collect all Edge Impulse SDK source files
file(GLOB_RECURSE CMSIS_DSP_SOURCES
    "edge-impulse-sdk/CMSIS/DSP/Source/BasicMathFunctions/*.c"
    "edge-impulse-sdk/CMSIS/DSP/Source/MatrixFunctions/*.c"
    "edge-impulse-sdk/CMSIS/DSP/Source/StatisticsFunctions/*.c"
    "edge-impulse-sdk/CMSIS/DSP/Source/TransformFunctions/*.c"
    "edge-impulse-sdk/CMSIS/DSP/Source/CommonTables/*.c"
    "edge-impulse-sdk/CMSIS/DSP/Source/SupportFunctions/*.c"
)

file(GLOB_RECURSE CMSIS_NN_SOURCES
    "edge-impulse-sdk/CMSIS/NN/Source/*.c"
)

file(GLOB_RECURSE TFLITE_SOURCES
    "edge-impulse-sdk/tensorflow/lite/c/*.c"
    "edge-impulse-sdk/tensorflow/lite/core/*.cc"
    "edge-impulse-sdk/tensorflow/lite/kernels/*.cc"
    "edge-impulse-sdk/tensorflow/lite/kernels/internal/*.cc"
    "edge-impulse-sdk/tensorflow/lite/micro/*.cc"
    "edge-impulse-sdk/tensorflow/lite/micro/kernels/*.cc"
    "edge-impulse-sdk/tensorflow/lite/micro/memory_planner/*.cc"
    "edge-impulse-sdk/tensorflow/lite/schema/*.cc"
)

file(GLOB_RECURSE EI_DSP_SOURCES
    "edge-impulse-sdk/dsp/*.cpp"
)

# No external porting needed - using custom ei_porting_esp32.cpp
set(EI_PORTING_SOURCES "")

set(EI_SDK_SOURCES ${CMSIS_DSP_SOURCES} ${CMSIS_NN_SOURCES} ${TFLITE_SOURCES} ${EI_DSP_SOURCES} ${EI_PORTING_SOURCES})

# Collect TFLite model files
file(GLOB TFLITE_MODEL_SOURCES
    "tflite-model/*.cpp"
    "tflite-model/*.cc"
)

# Main application source
set(MAIN_SOURCES
    "Fan_Anomaly_Detection.cpp"
    "ei_porting_esp32.cpp"
)

idf_component_register(
    SRCS ${MAIN_SOURCES} ${EI_SDK_SOURCES} ${TFLITE_MODEL_SOURCES}
    INCLUDE_DIRS
        "."
        "edge-impulse-sdk"
        "edge-impulse-sdk/tensorflow"
        "edge-impulse-sdk/third_party"
        "edge-impulse-sdk/third_party/flatbuffers/include"
        "edge-impulse-sdk/third_party/gemmlowp"
        "edge-impulse-sdk/third_party/ruy"
        "model-parameters"
        "tflite-model"
    REQUIRES driver nvs_flash esp_wifi esp_http_server esp_timer
)

# C++ compilation flags
target_compile_options(${COMPONENT_LIB} PRIVATE
    -std=gnu++14
    -fno-rtti
    -fno-exceptions
    -fno-threadsafe-statics
    -fmessage-length=0
    -DTF_LITE_STATIC_MEMORY
    -DTF_LITE_DISABLE_X86_NEON
    -DEI_CLASSIFIER_TFLITE_ENABLE_CMSIS_NN=1
    -DEIDSP_SIGNAL_C_FN_POINTER=0
    -DEI_C_LINKAGE=1
    -DEIDSP_USE_CMSIS_DSP=0
    -DDISABLE_ESP_DSP=1
    -Wno-maybe-uninitialized
    -Wno-sign-compare
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-format
    -Wno-unused-but-set-parameter
    -Wno-missing-field-initializers
)
