#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace pa {

// Logical layouts:
//   Q/O: [batch, query_head, head_dim]
//   K/V: [physical_page, token_in_page, kv_head, head_dim]
//   block_tables: [batch, max_pages_per_sequence]
struct Problem {
  int batch{};
  int query_heads{};
  int kv_heads{};
  int head_dim{};
  int page_size{};
  int max_pages_per_sequence{};

  [[nodiscard]] __host__ __device__ constexpr int queries_per_kv() const {
    return query_heads / kv_heads;
  }
  [[nodiscard]] __host__ __device__ constexpr std::size_t query_elements() const {
    return static_cast<std::size_t>(batch) * query_heads * head_dim;
  }
};

struct Inputs {
  float const* query{};
  float const* key_cache{};
  float const* value_cache{};
  int const* block_tables{};
  int const* context_lengths{};
  float scale{};
};

enum class KernelKind { Baseline, FusedV1, VectorizedV2, GqaReuseV3, SplitKvV4 };

}  // namespace pa
