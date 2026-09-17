#include "paged_attention/api.cuh"
#include "paged_attention/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Case {
  pa::Problem problem;
  int min_context;
  int max_context;
};

Case get_case(std::string const& name) {
  if (name == "tiny") return {{2, 8, 2, 64, 16, 4}, 7, 49};
  if (name == "interactive") return {{8, 32, 8, 128, 16, 128}, 128, 2048};
  if (name == "throughput") return {{64, 32, 8, 128, 16, 256}, 512, 4096};
  if (name == "long") return {{4, 32, 8, 128, 16, 4096}, 8192, 65536};
  if (name == "ragged") return {{64, 32, 8, 128, 16, 1024}, 16, 16384};
  throw std::runtime_error("unknown case: " + name);
}

pa::KernelKind parse_kernel(std::string const& name) {
  if (name == "baseline") return pa::KernelKind::Baseline;
  if (name == "fused_v1") return pa::KernelKind::FusedV1;
  if (name == "vectorized_v2") return pa::KernelKind::VectorizedV2;
  if (name == "gqa_reuse_v3") return pa::KernelKind::GqaReuseV3;
  if (name == "split_kv_v4") return pa::KernelKind::SplitKvV4;
  throw std::runtime_error("unknown kernel: " + name);
}

void reference(pa::Problem const& p, std::vector<float> const& q,
               std::vector<float> const& k, std::vector<float> const& v,
               std::vector<int> const& tables, std::vector<int> const& lengths,
               std::vector<float>& out) {
  float const scale = 1.0f / std::sqrt(static_cast<float>(p.head_dim));
  std::vector<float> scores;
  for (int b = 0; b < p.batch; ++b) for (int h = 0; h < p.query_heads; ++h) {
    int const kvh = h / p.queries_per_kv();
    scores.assign(lengths[b], 0.0f);
    float maximum = -INFINITY;
    for (int t = 0; t < lengths[b]; ++t) {
      int page = tables[b * p.max_pages_per_sequence + t / p.page_size];
      std::size_t kb = ((static_cast<std::size_t>(page) * p.page_size + t % p.page_size) * p.kv_heads + kvh) * p.head_dim;
      std::size_t qb = (static_cast<std::size_t>(b) * p.query_heads + h) * p.head_dim;
      for (int d = 0; d < p.head_dim; ++d) scores[t] += q[qb + d] * k[kb + d];
      scores[t] *= scale;
      maximum = std::max(maximum, scores[t]);
    }
    float sum = 0.0f;
    for (float score : scores) sum += std::exp(score - maximum);
    for (int t = 0; t < lengths[b]; ++t) {
      int page = tables[b * p.max_pages_per_sequence + t / p.page_size];
      std::size_t vb = ((static_cast<std::size_t>(page) * p.page_size + t % p.page_size) * p.kv_heads + kvh) * p.head_dim;
      float weight = std::exp(scores[t] - maximum) / sum;
      std::size_t ob = (static_cast<std::size_t>(b) * p.query_heads + h) * p.head_dim;
      for (int d = 0; d < p.head_dim; ++d) out[ob + d] += weight * v[vb + d];
    }
  }
}

template <class T> std::vector<T> random_values(std::size_t n, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
  std::vector<T> values(n);
  for (auto& x : values) x = static_cast<T>(dist(rng));
  return values;
}
}  // namespace

int main(int argc, char** argv) try {
  std::string case_name = "tiny", kernel_name = "baseline";
  int iterations = 50;
  int fixed_context = 0;
  bool check = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--case" && i + 1 < argc) case_name = argv[++i];
    else if (arg == "--kernel" && i + 1 < argc) kernel_name = argv[++i];
    else if (arg == "--iterations" && i + 1 < argc) iterations = std::stoi(argv[++i]);
    else if (arg == "--context" && i + 1 < argc) fixed_context = std::stoi(argv[++i]);
    else if (arg == "--check") check = true;
    else throw std::runtime_error("usage: pa_benchmark [--case tiny|interactive|throughput|long|ragged] [--kernel baseline|fused_v1|vectorized_v2|gqa_reuse_v3|split_kv_v4] [--iterations N] [--context N] [--check]");
  }

  Case config = get_case(case_name);
  pa::Problem const& p = config.problem;
  if (p.query_heads % p.kv_heads != 0 || (p.head_dim != 64 && p.head_dim != 128))
    throw std::runtime_error("invalid problem shape");

  std::mt19937 rng(2026);
  std::uniform_int_distribution<int> length_dist(config.min_context, config.max_context);
  std::vector<int> lengths(p.batch);
  int total_pages = 0;
  if (fixed_context < 0 || fixed_context > p.max_pages_per_sequence * p.page_size)
    throw std::runtime_error("--context exceeds this case's page-table capacity");
  for (int& length : lengths) {
    length = fixed_context ? fixed_context : length_dist(rng);
    total_pages += (length + p.page_size - 1) / p.page_size;
  }
  std::vector<int> physical_pages(total_pages);
  std::iota(physical_pages.begin(), physical_pages.end(), 0);
  std::shuffle(physical_pages.begin(), physical_pages.end(), rng);
  std::vector<int> tables(static_cast<std::size_t>(p.batch) * p.max_pages_per_sequence, 0);
  int cursor = 0;
  for (int b = 0; b < p.batch; ++b) {
    int pages = (lengths[b] + p.page_size - 1) / p.page_size;
    for (int j = 0; j < pages; ++j) tables[b * p.max_pages_per_sequence + j] = physical_pages[cursor++];
  }

  auto q = random_values<float>(p.query_elements(), 1);
  std::size_t cache_elements = static_cast<std::size_t>(total_pages) * p.page_size * p.kv_heads * p.head_dim;
  auto k = random_values<float>(cache_elements, 2);
  auto v = random_values<float>(cache_elements, 3);
  std::vector<float> output(p.query_elements()), expected(p.query_elements(), 0.0f);
  pa::DeviceBuffer<float> dq(q.size()), dk(k.size()), dv(v.size()), dout(output.size());
  pa::DeviceBuffer<int> dtables(tables.size()), dlengths(lengths.size());
  dq.upload(q.data()); dk.upload(k.data()); dv.upload(v.data());
  dtables.upload(tables.data()); dlengths.upload(lengths.data());
  pa::Inputs inputs{dq.get(), dk.get(), dv.get(), dtables.get(), dlengths.get(),
                    1.0f / std::sqrt(static_cast<float>(p.head_dim))};
  auto kind = parse_kernel(kernel_name);

  for (int i = 0; i < 10; ++i) pa::launch(kind, p, inputs, dout.get());
  PA_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  PA_CUDA_CHECK(cudaEventCreate(&start)); PA_CUDA_CHECK(cudaEventCreate(&stop));
  PA_CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) pa::launch(kind, p, inputs, dout.get());
  PA_CUDA_CHECK(cudaEventRecord(stop)); PA_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  PA_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  PA_CUDA_CHECK(cudaGetLastError());
  std::cout << std::fixed << std::setprecision(3) << "case=" << case_name
            << " kernel=" << kernel_name << " mean_us="
            << elapsed_ms * 1000.0f / iterations << '\n';

  if (check) {
    if (case_name != "tiny") std::cerr << "warning: CPU reference may be slow for this case\n";
    reference(p, q, k, v, tables, lengths, expected);
    dout.download(output.data());
    int errors = 0; float worst = 0.0f;
    for (std::size_t i = 0; i < output.size(); ++i) {
      float error = std::abs(output[i] - expected[i]);
      worst = std::max(worst, error);
      if (!std::isfinite(output[i]) || error > 2e-4f + 2e-4f * std::abs(expected[i])) ++errors;
    }
    std::cout << "check=" << (errors ? "FAIL" : "PASS") << " errors=" << errors
              << " worst_abs_error=" << worst << '\n';
    if (errors) return EXIT_FAILURE;
  }
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return EXIT_SUCCESS;
} catch (std::exception const& error) {
  std::cerr << "error: " << error.what() << '\n';
  return EXIT_FAILURE;
}
