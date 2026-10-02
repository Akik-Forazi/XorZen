#pragma once

#include <exception>
#include <string>
#include <sstream>
#include <unordered_map>

namespace xorzen {

class XorzenError : public std::runtime_error {
public:
    XorzenError(const std::string& message, 
                const std::unordered_map<std::string, std::string>& details = {},
                const std::string& suggestion = "")
        : std::runtime_error(format_message(message, details, suggestion)),
          message_(message), details_(details), suggestion_(suggestion) {}

    const std::string& get_message() const { return message_; }
    const std::unordered_map<std::string, std::string>& get_details() const { return details_; }
    const std::string& get_suggestion() const { return suggestion_; }

private:
    std::string message_;
    std::unordered_map<std::string, std::string> details_;
    std::string suggestion_;

    static std::string format_message(const std::string& msg, 
                                      const std::unordered_map<std::string, std::string>& details,
                                      const std::string& suggestion) {
        std::stringstream ss;
        ss << "XorzenError: " << msg;
        if (!details.empty()) {
            ss << "
Details: ";
            for (auto const& [k, v] : details) ss << k << "=" << v << " ";
        }
        if (!suggestion.empty()) ss << "
💡 Suggestion: " << suggestion;
        return ss.str();
    }
};

// Domain-Specific Errors
class ModelError : public XorzenError { using XorzenError::XorzenError; };
class DataError : public XorzenError { using XorzenError::XorzenError; };
class TokenizerError : public XorzenError { using XorzenError::XorzenError; };
class TrainingError : public XorzenError { using XorzenError::XorzenError; };

} // namespace xorzen
