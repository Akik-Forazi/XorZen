#pragma once

#include <iostream>
#include <string>
#include <chrono>
#include <torch/torch.h>

namespace xorzen {

enum class LogLevel { DEBUG, INFO, WARNING, ERROR, CRITICAL };

class Logger {
public:
    static void set_level(LogLevel level) { min_level = level; }
    
    static void log(LogLevel level, const std::string& module, const std::string& message) {
        if (level < min_level) return;
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << "[" << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ") << "] "
                  << "[" << level_to_string(level) << "] "
                  << "[" << module << "] " << message << std::endl;
    }

private:
    static LogLevel min_level;
    static std::string level_to_string(LogLevel level) {
        switch(level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::CRITICAL: return "CRITICAL";
            default: return "INFO";
        }
    }
};

// Debug macro for tensor inspection
#define XZ_DEBUG_TENSOR(tensor, name) \
    if (xorzen::Logger::min_level <= xorzen::LogLevel::DEBUG) { \
        std::cout << "[DEBUG] [TENSOR] " << name << ": " << tensor.sizes() \
                  << " mean=" << tensor.mean().item<float>() \
                  << " std=" << tensor.std().item<float>() << std::endl; \
    }

} // namespace xorzen
