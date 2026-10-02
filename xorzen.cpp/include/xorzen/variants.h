#pragma once
// ============================================================
//  xorzen.cpp — include/xorzen/variants.h
//  ZARX Model Size Variants
//  Ported from xorzen/models/zero/variants.py
//  FRAZIYM TECH & AI / Akik Faraji
// ============================================================
// Precision-tuned parameter targets from ZARX_SCALING_LAWS.md:
//   TINY_23K:  37,824 params
//   NANO_1M:   1,077,503 params
//   NANO_10M:  10,970,548 params
//   MICRO_50M: 50,399,371 params
//   MINI_277M: 277,000,335 params
//   SMALL_500M:500,000,083 params
//   MEDIUM_1B: 1,000,000,886 params
//   XL_7B:     7,000,000,466 params
// ============================================================

#include "xorzen/types.h"
#include <string>

namespace xorzen {

// ============================================================
//  ModelSize enum
// ============================================================
enum class ModelSize {
    TINY_23K,
    NANO_1M,
    NANO_10M,
    MICRO_50M,
    MINI_277M,
    SMALL_500M,
    MEDIUM_1B,
    LARGE_3B,   // Python: LARGE_3B (was XL_3B — renamed for parity)
    XL_7B,
    XXL_13B,    // Python: XXL_13B
    XXXL_70B,   // Python: XXXL_70B
    GREED_TINY, // Python: GREED_TINY
    GREED_SMALL // Python: GREED_SMALL
};

// ============================================================
//  ConfigFactory — builds precision-tuned ModelConfig per size
// ============================================================
class ConfigFactory {
public:
    static ModelConfig get_config(ModelSize size);
    static std::string size_name(ModelSize size);
    static int64_t     param_target(ModelSize size);
};

// ============================================================
//  Convenience factory functions
// ============================================================
inline ModelConfig tiny_23k_config()   { return ConfigFactory::get_config(ModelSize::TINY_23K);   }
inline ModelConfig nano_1m_config()    { return ConfigFactory::get_config(ModelSize::NANO_1M);    }
inline ModelConfig nano_10m_config()   { return ConfigFactory::get_config(ModelSize::NANO_10M);   }
inline ModelConfig micro_50m_config()  { return ConfigFactory::get_config(ModelSize::MICRO_50M);  }
inline ModelConfig mini_277m_config()  { return ConfigFactory::get_config(ModelSize::MINI_277M);  }
inline ModelConfig small_500m_config() { return ConfigFactory::get_config(ModelSize::SMALL_500M); }
inline ModelConfig medium_1b_config()  { return ConfigFactory::get_config(ModelSize::MEDIUM_1B);  }
inline ModelConfig large_3b_config()   { return ConfigFactory::get_config(ModelSize::LARGE_3B);   }
inline ModelConfig xl_7b_config()      { return ConfigFactory::get_config(ModelSize::XL_7B);      }

} // namespace xorzen
