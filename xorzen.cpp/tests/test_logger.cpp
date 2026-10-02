#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "xorzen/utils/logger.h"

TEST_CASE("Logger functional test", "[logger]") {
    xorzen::Logger::set_level(xorzen::LogLevel::DEBUG);
    
    // Just ensuring it doesn't crash
    xorzen::Logger::log(xorzen::LogLevel::INFO, "test", "Testing logger...");
    SUCCEED();
}
