/**
 * Fan Anomaly Detection using Edge Impulse + INMP441 + SSD1306
 *
 * Hardware:
 * - ESP32
 * - INMP441 I2S Microphone (16kHz)
 * - SSD1306 OLED Display (128x64, I2C)
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// FreeRTOS
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// ESP32 Drivers
#include "driver/i2s_std.h"
#include "driver/i2c_master.h"
#include "driver/gpio.h"
#include "hal/i2c_types.h"
#include "soc/clk_tree_defs.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"  // High-resolution timer for accurate timing

// Edge Impulse
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"

static const char *TAG = "FAN_ANOMALY";

// ==================== I2S Configuration (INMP441) ====================
#define I2S_WS_PIN      GPIO_NUM_15  // Word Select (LRCL)
#define I2S_SD_PIN      GPIO_NUM_32  // Serial Data (DOUT)
#define I2S_SCK_PIN     GPIO_NUM_14  // Serial Clock (BCLK)

#define I2S_SAMPLE_RATE 16000        // Match training data
#define I2S_SAMPLE_BITS I2S_DATA_BIT_WIDTH_32BIT
#define I2S_DMA_BUF_LEN 512

// ==================== I2C Configuration (OLED) ====================
static const i2c_port_num_t i2c_port = 0;
static const gpio_num_t i2c_sda_pin = GPIO_NUM_21;
static const gpio_num_t i2c_scl_pin = GPIO_NUM_22;
static const uint8_t i2c_glitch_ignore_cnt = 7;
static const uint16_t oled_addr = 0x3C;
static const uint32_t oled_scl_speed_hz = 100000;

#define OLED_WIDTH 128
#define OLED_HEIGHT 64
#define OLED_BUFFER_SIZE (OLED_WIDTH * OLED_HEIGHT / 8)

// ==================== Edge Impulse Configuration ====================
#define EI_CLASSIFIER_RAW_SAMPLE_COUNT 416  // From model_metadata.h
#define EI_CLASSIFIER_FREQUENCY 16000

// ==================== Global Variables ====================
static i2s_chan_handle_t rx_handle = NULL;
static i2c_master_dev_handle_t oled_dev;
static uint8_t oled_buffer[OLED_BUFFER_SIZE];

// Audio buffer for inference
static int16_t inference_buffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT];
static bool audio_ready = false;

// Performance metrics
typedef struct {
    int64_t audio_capture_us;      // Time to capture audio buffer
    int64_t inference_us;           // Time for model inference
    int64_t display_update_us;      // Time to update display
    int64_t total_cycle_us;         // End-to-end cycle time
    float fps;                       // Inferences per second
    uint32_t inference_count;       // Total inferences performed
} performance_metrics_t;

static performance_metrics_t perf_metrics = {0};
static int64_t audio_capture_start_time = 0;

// Simple 5x7 font for displaying text
const uint8_t font5x7[][5] = {
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, // A
    {0x7F, 0x49, 0x49, 0x49, 0x36}, // B
    {0x3E, 0x41, 0x41, 0x41, 0x22}, // C
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, // D
    {0x7F, 0x49, 0x49, 0x49, 0x41}, // E
    {0x7F, 0x09, 0x09, 0x09, 0x01}, // F
    {0x3E, 0x41, 0x49, 0x49, 0x7A}, // G
    {0x7F, 0x08, 0x08, 0x08, 0x7F}, // H
    {0x00, 0x41, 0x7F, 0x41, 0x00}, // I
    {0x20, 0x40, 0x41, 0x3F, 0x01}, // J
    {0x7F, 0x08, 0x14, 0x22, 0x41}, // K
    {0x7F, 0x40, 0x40, 0x40, 0x40}, // L
    {0x7F, 0x02, 0x0C, 0x02, 0x7F}, // M
    {0x7F, 0x04, 0x08, 0x10, 0x7F}, // N
    {0x3E, 0x41, 0x41, 0x41, 0x3E}, // O
    {0x7F, 0x09, 0x09, 0x09, 0x06}, // P
    {0x3E, 0x41, 0x51, 0x21, 0x5E}, // Q
    {0x7F, 0x09, 0x19, 0x29, 0x46}, // R
    {0x46, 0x49, 0x49, 0x49, 0x31}, // S
    {0x01, 0x01, 0x7F, 0x01, 0x01}, // T
    {0x3F, 0x40, 0x40, 0x40, 0x3F}, // U
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, // V
    {0x3F, 0x40, 0x38, 0x40, 0x3F}, // W
    {0x63, 0x14, 0x08, 0x14, 0x63}, // X
    {0x07, 0x08, 0x70, 0x08, 0x07}, // Y
    {0x61, 0x51, 0x49, 0x45, 0x43}, // Z
    {0x00, 0x00, 0x00, 0x00, 0x00}, // Space
    {0x00, 0x00, 0x5F, 0x00, 0x00}, // !
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, // 0
    {0x00, 0x42, 0x7F, 0x40, 0x00}, // 1
    {0x42, 0x61, 0x51, 0x49, 0x46}, // 2
    {0x21, 0x41, 0x45, 0x4B, 0x31}, // 3
    {0x18, 0x14, 0x12, 0x7F, 0x10}, // 4
    {0x27, 0x45, 0x45, 0x45, 0x39}, // 5
    {0x3C, 0x4A, 0x49, 0x49, 0x30}, // 6
    {0x01, 0x71, 0x09, 0x05, 0x03}, // 7
    {0x36, 0x49, 0x49, 0x49, 0x36}, // 8
    {0x06, 0x49, 0x49, 0x29, 0x1E}, // 9
    {0x00, 0x36, 0x36, 0x00, 0x00}, // :
};

// ==================== OLED Functions ====================
static esp_err_t oled_write_command(uint8_t cmd) {
    uint8_t buf[2] = {0x00, cmd};
    return i2c_master_transmit(oled_dev, buf, sizeof(buf), -1);
}

static void oled_clear_buffer(void) {
    memset(oled_buffer, 0x00, OLED_BUFFER_SIZE);
}

static void oled_fill_buffer(void) {
    memset(oled_buffer, 0xFF, OLED_BUFFER_SIZE);
}

static void oled_set_pixel(int x, int y, bool on) {
    if (x < 0 || x >= OLED_WIDTH || y < 0 || y >= OLED_HEIGHT) return;
    int byte_index = x + (y / 8) * OLED_WIDTH;
    int bit_index = y % 8;
    if (on) {
        oled_buffer[byte_index] |= (1 << bit_index);
    } else {
        oled_buffer[byte_index] &= ~(1 << bit_index);
    }
}

static void oled_draw_char(int x, int y, char c, bool on) {
    int index = -1;
    if (c >= 'A' && c <= 'Z') index = c - 'A';
    else if (c >= 'a' && c <= 'z') index = c - 'a';
    else if (c == ' ') index = 26;
    else if (c == '!') index = 27;
    else if (c >= '0' && c <= '9') index = 28 + (c - '0');
    else if (c == ':') index = 38;

    if (index < 0) return;

    for (int i = 0; i < 5; i++) {
        uint8_t col = font5x7[index][i];
        for (int j = 0; j < 7; j++) {
            if (col & (1 << j)) {
                oled_set_pixel(x + i, y + j, on);
            }
        }
    }
}

static void oled_draw_string(int x, int y, const char *str, bool on) {
    while (*str) {
        oled_draw_char(x, y, *str, on);
        x += 6;
        str++;
    }
}

static void oled_update(void) {
    for (uint8_t page = 0; page < 8; page++) {
        oled_write_command(0xB0 + page);
        oled_write_command(0x00);
        oled_write_command(0x10);

        uint8_t buf[1 + OLED_WIDTH];
        buf[0] = 0x40;
        for (int i = 0; i < OLED_WIDTH; i++) {
            buf[1 + i] = oled_buffer[OLED_WIDTH * page + i];
        }
        i2c_master_transmit(oled_dev, buf, sizeof(buf), -1);
    }
}

static void oled_init(void) {
    oled_write_command(0xAE); // Display off
    oled_write_command(0x20); // Set Memory Addressing Mode
    oled_write_command(0x00); // Horizontal Addressing Mode
    oled_write_command(0xB0); // Set Page Start Address
    oled_write_command(0xC8); // COM Output Scan Direction
    oled_write_command(0x00); // Low column address
    oled_write_command(0x10); // High column address
    oled_write_command(0x40); // Start line address
    oled_write_command(0x81); // Contrast control
    oled_write_command(0x7F);
    oled_write_command(0xA1); // Segment re-map
    oled_write_command(0xA6); // Normal display
    oled_write_command(0xA8); // Multiplex ratio
    oled_write_command(0x3F); // 1/64 duty
    oled_write_command(0xA4); // Display follows RAM
    oled_write_command(0xD3); // Display offset
    oled_write_command(0x00);
    oled_write_command(0xD5); // Display clock divide
    oled_write_command(0x80);
    oled_write_command(0xD9); // Pre-charge period
    oled_write_command(0xF1);
    oled_write_command(0xDA); // COM pins hardware config
    oled_write_command(0x12);
    oled_write_command(0xDB); // VCOMH deselect level
    oled_write_command(0x40);
    oled_write_command(0x8D); // Charge pump
    oled_write_command(0x14);
    oled_write_command(0xAF); // Display on
}

// ==================== I2S Functions ====================
static void i2s_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 4;
    chan_cfg.dma_frame_num = I2S_DMA_BUF_LEN;

    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(I2S_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_SAMPLE_BITS, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_SCK_PIN,
            .ws = I2S_WS_PIN,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_SD_PIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));

    ESP_LOGI(TAG, "I2S initialized @ %d Hz", I2S_SAMPLE_RATE);
}

// ==================== Audio Capture Task ====================
static void audio_capture_task(void *pvParameters) {
    size_t bytes_read;
    int32_t *i2s_buffer = (int32_t *)malloc(I2S_DMA_BUF_LEN * sizeof(int32_t));
    int buffer_index = 0;

    ESP_LOGI(TAG, "Audio capture started, collecting %d samples", EI_CLASSIFIER_RAW_SAMPLE_COUNT);

    while (1) {
        // Mark start of audio capture when buffer is empty
        if (buffer_index == 0) {
            audio_capture_start_time = esp_timer_get_time();
        }

        esp_err_t ret = i2s_channel_read(rx_handle, i2s_buffer,
                                         I2S_DMA_BUF_LEN * sizeof(int32_t),
                                         &bytes_read, portMAX_DELAY);

        if (ret == ESP_OK) {
            int samples_read = bytes_read / sizeof(int32_t);

            for (int i = 0; i < samples_read && buffer_index < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
                // Convert 32-bit I2S sample to 16-bit
                inference_buffer[buffer_index++] = (int16_t)(i2s_buffer[i] >> 14);

                if (buffer_index >= EI_CLASSIFIER_RAW_SAMPLE_COUNT) {
                    // Calculate audio capture time
                    perf_metrics.audio_capture_us = esp_timer_get_time() - audio_capture_start_time;
                    audio_ready = true;
                    buffer_index = 0;
                }
            }
        }

        vTaskDelay(1);
    }

    free(i2s_buffer);
    vTaskDelete(NULL);
}

// ==================== Inference Task ====================
static void inference_task(void *pvParameters) {
    ESP_LOGI(TAG, "Inference task started");
    ESP_LOGI(TAG, "┌────────────────────────────────────────────────────────┐");
    ESP_LOGI(TAG, "│         PERFORMANCE MONITORING ENABLED                 │");
    ESP_LOGI(TAG, "│ Timing measurements in microseconds (μs)              │");
    ESP_LOGI(TAG, "└────────────────────────────────────────────────────────┘");

    signal_t signal;
    ei_impulse_result_t result = {0};
    int64_t cycle_start_time = 0;
    int64_t last_report_time = esp_timer_get_time();

    while (1) {
        if (!audio_ready) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        // Start total cycle timer
        cycle_start_time = esp_timer_get_time();

        // Prepare signal
        signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
        signal.get_data = [](size_t offset, size_t length, float *out_ptr) -> int {
            for (size_t i = 0; i < length; i++) {
                out_ptr[i] = (float)inference_buffer[offset + i];
            }
            return 0;
        };

        audio_ready = false;

        // === MEASURE INFERENCE TIME ===
        int64_t inference_start = esp_timer_get_time();
        EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
        int64_t inference_end = esp_timer_get_time();

        perf_metrics.inference_us = inference_end - inference_start;

        if (res != EI_IMPULSE_OK) {
            ESP_LOGE(TAG, "Inference failed: %d", res);
            continue;
        }

        // Get classification results
        float normal_score = 0.0f;
        float anomaly_score = 0.0f;

        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            const char *label = result.classification[ix].label;
            float value = result.classification[ix].value;

            if (strcmp(label, "normal") == 0 || strcmp(label, "Normal") == 0) {
                normal_score = value;
            } else if (strcmp(label, "abnormal") == 0 || strcmp(label, "Abnormal") == 0) {
                anomaly_score = value;
            }
        }

        // === MEASURE DISPLAY UPDATE TIME ===
        int64_t display_start = esp_timer_get_time();

        oled_clear_buffer();

        // Title
        oled_draw_string(10, 5, "FAN STATUS", true);

        // Draw horizontal line
        for (int x = 0; x < OLED_WIDTH; x++) {
            oled_set_pixel(x, 16, true);
        }

        // Result
        if (anomaly_score > normal_score) {
            oled_draw_string(15, 25, "ANOMALY!", true);

            // Draw warning box
            for (int x = 10; x < OLED_WIDTH - 10; x++) {
                oled_set_pixel(x, 22, true);
                oled_set_pixel(x, 38, true);
            }
            for (int y = 22; y < 38; y++) {
                oled_set_pixel(10, y, true);
                oled_set_pixel(OLED_WIDTH - 11, y, true);
            }
        } else {
            oled_draw_string(20, 25, "NORMAL", true);
        }

        // Show scores
        char score_text[32];
        snprintf(score_text, sizeof(score_text), "N:%.0f A:%.0f", normal_score * 100, anomaly_score * 100);
        oled_draw_string(20, 45, score_text, true);

        // Show inference time on display
        char timing_text[32];
        snprintf(timing_text, sizeof(timing_text), "%lldms", perf_metrics.inference_us / 1000);
        oled_draw_string(5, 55, timing_text, true);

        oled_update();

        int64_t display_end = esp_timer_get_time();
        perf_metrics.display_update_us = display_end - display_start;

        // Calculate total cycle time
        perf_metrics.total_cycle_us = esp_timer_get_time() - cycle_start_time;
        perf_metrics.inference_count++;
        perf_metrics.fps = 1000000.0f / perf_metrics.total_cycle_us;

        // === DETAILED PERFORMANCE LOGGING ===
        // Log every inference with detailed breakdown
        ESP_LOGI(TAG, "─────────────────────────────────────────────────");
        ESP_LOGI(TAG, "Inference #%lu | Result: %s (%.1f%%)",
                 perf_metrics.inference_count,
                 (anomaly_score > normal_score) ? "ANOMALY" : "NORMAL",
                 (anomaly_score > normal_score ? anomaly_score : normal_score) * 100);
        ESP_LOGI(TAG, "  Audio Capture:  %5lld μs (%4.1f ms)",
                 perf_metrics.audio_capture_us,
                 perf_metrics.audio_capture_us / 1000.0f);
        ESP_LOGI(TAG, "  Inference Time: %5lld μs (%4.1f ms) ⚡",
                 perf_metrics.inference_us,
                 perf_metrics.inference_us / 1000.0f);
        ESP_LOGI(TAG, "  Display Update: %5lld μs (%4.1f ms)",
                 perf_metrics.display_update_us,
                 perf_metrics.display_update_us / 1000.0f);
        ESP_LOGI(TAG, "  Total Cycle:    %5lld μs (%4.1f ms)",
                 perf_metrics.total_cycle_us,
                 perf_metrics.total_cycle_us / 1000.0f);
        ESP_LOGI(TAG, "  Throughput:     %.2f inferences/sec", perf_metrics.fps);

        // Print summary statistics every 10 seconds
        int64_t current_time = esp_timer_get_time();
        if ((current_time - last_report_time) >= 10000000) {  // 10 seconds
            ESP_LOGI(TAG, "");
            ESP_LOGI(TAG, "╔════════════════════════════════════════════════════╗");
            ESP_LOGI(TAG, "║          10-SECOND PERFORMANCE SUMMARY             ║");
            ESP_LOGI(TAG, "╠════════════════════════════════════════════════════╣");
            ESP_LOGI(TAG, "║ Total Inferences:     %5lu                        ║", perf_metrics.inference_count);
            ESP_LOGI(TAG, "║ Avg Inference Time:   %4.1f ms                     ║", perf_metrics.inference_us / 1000.0f);
            ESP_LOGI(TAG, "║ Avg Total Latency:    %4.1f ms                     ║", perf_metrics.total_cycle_us / 1000.0f);
            ESP_LOGI(TAG, "║ Throughput:           %4.1f inferences/sec         ║", perf_metrics.fps);
            ESP_LOGI(TAG, "╚════════════════════════════════════════════════════╝");
            ESP_LOGI(TAG, "");
            last_report_time = current_time;
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }

    vTaskDelete(NULL);
}

// ==================== Main ====================
extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Fan Anomaly Detection Starting");
    ESP_LOGI(TAG, "Model: %d sample input, %d labels", EI_CLASSIFIER_RAW_SAMPLE_COUNT, EI_CLASSIFIER_LABEL_COUNT);

    // Initialize I2C for OLED
    i2c_master_bus_handle_t i2c_bus;
    i2c_master_bus_config_t bus_config = {
        .i2c_port = i2c_port,
        .sda_io_num = i2c_sda_pin,
        .scl_io_num = i2c_scl_pin,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = i2c_glitch_ignore_cnt,
        .flags = {
            .enable_internal_pullup = true,
        },
    };

    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_config, &i2c_bus));

    i2c_device_config_t dev_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = oled_addr,
        .scl_speed_hz = oled_scl_speed_hz,
    };

    ESP_ERROR_CHECK(i2c_master_bus_add_device(i2c_bus, &dev_config, &oled_dev));

    oled_init();
    ESP_LOGI(TAG, "OLED initialized");

    // Show startup screen
    oled_clear_buffer();
    oled_draw_string(5, 10, "FAN ANOMALY", true);
    oled_draw_string(10, 25, "DETECTION", true);
    oled_draw_string(15, 40, "LOADING", true);
    oled_update();

    vTaskDelay(pdMS_TO_TICKS(2000));

    // Initialize I2S
    i2s_init();

    // Start tasks
    xTaskCreatePinnedToCore(audio_capture_task, "audio_capture", 4096, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(inference_task, "inference", 8192, NULL, 4, NULL, 0);

    ESP_LOGI(TAG, "System initialized successfully!");
}
