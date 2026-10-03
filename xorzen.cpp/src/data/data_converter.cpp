#include "xorzen/data_pipeline.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

namespace xorzen {

DataConverter::DataConverter(BEBPETokenizer tokenizer) : tokenizer_(std::move(tokenizer)) {}

DataConverter::ConversionStats DataConverter::txt_to_bin(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    int64_t max_length,
    int64_t stride,
    bool add_special_tokens
) {
    std::ifstream f_in(input_path);
    if (!f_in.is_open()) {
        throw std::runtime_error("Could not open input file: " + input_path.string());
    }

    ConversionStats stats;
    std::vector<int64_t> all_tokens;
    std::string line;

    while (std::getline(f_in, line)) {
        if (line.empty()) continue;
        stats.lines_read++;

        auto encoding = tokenizer_.encode(line, add_special_tokens);
        all_tokens.insert(all_tokens.end(), encoding.input_ids.begin(), encoding.input_ids.end());
    }

    // Determine optimal dtype (uint16 if vocab < 65536)
    bool use_uint16 = tokenizer_.vocab_size() <= 65535;
    
    std::ofstream f_out(output_path, std::ios::binary);
    if (!f_out.is_open()) {
        throw std::runtime_error("Could not open output file: " + output_path.string());
    }

    if (max_length > 0) {
        // Create training records of length max_length + 1 so labels can be shifted by one.
        const int64_t record_length = max_length + 1;
        if (stride <= 0) stride = max_length;

        for (size_t i = 0; i + static_cast<size_t>(record_length) <= all_tokens.size();
             i += static_cast<size_t>(stride)) {
            for (int64_t j = 0; j < record_length; ++j) {
                int64_t token = all_tokens[i + j];
                if (use_uint16) {
                    uint16_t t16 = static_cast<uint16_t>(token);
                    f_out.write(reinterpret_cast<const char*>(&t16), sizeof(uint16_t));
                } else {
                    uint32_t t32 = static_cast<uint32_t>(token);
                    f_out.write(reinterpret_cast<const char*>(&t32), sizeof(uint32_t));
                }
            }
            stats.sequences_created++;
            stats.tokens_written += static_cast<size_t>(record_length);
        }
    } else {
        // Continuous stream
        for (int64_t token : all_tokens) {
            if (use_uint16) {
                uint16_t t16 = static_cast<uint16_t>(token);
                f_out.write(reinterpret_cast<const char*>(&t16), sizeof(uint16_t));
            } else {
                uint32_t t32 = static_cast<uint32_t>(token);
                f_out.write(reinterpret_cast<const char*>(&t32), sizeof(uint32_t));
            }
        }
        stats.tokens_written = all_tokens.size();
        stats.sequences_created = 1;
    }

    f_out.close();
    stats.file_size_bytes = std::filesystem::file_size(output_path);

    // Save metadata in a tiny schema understood by MMapDataset.
    std::ofstream f_meta(output_path.string() + ".meta.json");
    f_meta << "{\n"
           << "  \"dtype\": \"" << (use_uint16 ? "uint16" : "uint32") << "\",\n"
           << "  \"total_tokens\": " << stats.tokens_written << ",\n"
           << "  \"sequences\": " << stats.sequences_created << ",\n"
           << "  \"max_length\": " << max_length << ",\n"
           << "  \"record_length\": " << (max_length > 0 ? max_length + 1 : 0) << ",\n"
           << "  \"stride\": " << stride << ",\n"
           << "  \"vocab_size\": " << tokenizer_.vocab_size() << "\n"
           << "}\n";
    f_meta.close();

    return stats;
}

} // namespace xorzen
