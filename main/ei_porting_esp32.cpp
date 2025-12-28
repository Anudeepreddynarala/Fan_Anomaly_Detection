/**
 * Simple Edge Impulse porting layer for ESP32/ESP-IDF
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_log.h"

#define EI_WEAK __attribute__((weak))
#define EI_IMPULSE_ERROR int

extern "C" {

// Memory management
EI_WEAK void *ei_malloc(size_t size) {
    return malloc(size);
}

EI_WEAK void *ei_calloc(size_t nitems, size_t size) {
    return calloc(nitems, size);
}

EI_WEAK void ei_free(void *ptr) {
    free(ptr);
}

// Timing
EI_WEAK uint64_t ei_read_timer_us() {
    return esp_timer_get_time();
}

EI_WEAK uint64_t ei_read_timer_ms() {
    return esp_timer_get_time() / 1000;
}

// Logging
EI_WEAK void ei_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

EI_WEAK void ei_printf_float(float f) {
    printf("%f", f);
}

// Sleep
EI_WEAK void ei_sleep(uint32_t time_ms) {
    vTaskDelay(pdMS_TO_TICKS(time_ms));
}

// Cancelation check (not used in our case)
EI_WEAK EI_IMPULSE_ERROR ei_run_impulse_check_canceled() {
    return 0; // Never canceled
}

EI_WEAK EI_IMPULSE_ERROR ei_can_invoke_impulse() {
    return 0; // Always can invoke
}

} // extern "C"
