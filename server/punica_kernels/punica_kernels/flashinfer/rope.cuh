/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_ROPE_CUH_
#define FLASHINFER_ROPE_CUH_

#include <string>

namespace flashinfer {

/*!
 * \brief An enumeration class that defines different modes for applying RoPE
 * (Rotary Positional Embeddings).
 */
enum class RotaryMode {
  // No rotary positional embeddings
  kNone = 0U,
  // Apply Llama-style rope.
  kLlama = 1U,
};

/*!
 * \brief Convert RotaryMode to string
 * \param rotary_mode A RotaryMode value
 */
inline std::string RotaryModeToString(const RotaryMode& rotary_mode) {
  switch (rotary_mode) {
    case RotaryMode::kNone:
      return "None";
    case RotaryMode::kLlama:
      return "Llama";
    default:
      return "Unknown";
  }
}

}  // namespace flashinfer

#endif  // FLASHINFER_ROPE_CUH_