/* Copyright (c) 2021-2024, InterDigital Communications, Inc
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * * Neither the name of InterDigital Communications, Inc nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "rans_interface.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h> // PyTorch C++拡張機能

#include <iostream>
#include <algorithm>
#include <array>
#include <cassert>
#include <tuple>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <x86intrin.h>
#include "avx_mathfun.h" // これらのヘッダーはローカルに存在すると仮定します
#include "rans64.h"      // 同上

namespace py = pybind11;

/* probability range, this could be a parameter... */
constexpr int precision = 16;
constexpr int32_t max_cdf_value = 65535;
constexpr float offset = 0.5;
constexpr uint16_t bypass_precision = 4; /* number of bits in bypass mode */
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;
constexpr int K_gmm_default = 4; // GMMコンポーネントのデフォルト数 (pybindで使用)

namespace {

/* We only run this in debug mode as its costly... */
void assert_cdfs(const std::vector<std::vector<int>> &cdfs,
                 const std::vector<int> &cdfs_sizes) {
  for (int i = 0; i < static_cast<int>(cdfs.size()); ++i) {
    assert(cdfs[i][0] == 0);
    assert(cdfs[i][cdfs_sizes[i] - 1] == (1 << precision));
    for (int j = 0; j < cdfs_sizes[i] - 1; ++j) {
      assert(cdfs[i][j + 1] > cdfs[i][j]);
    }
  }
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

__attribute__((force_inline))
__m256 copysign_ps(__m256 from, __m256 to) {
    constexpr float signbit = -0.f;
    const __m256 avx_signbit = _mm256_broadcast_ss(&signbit);
    return _mm256_or_ps(_mm256_and_ps(avx_signbit, from), _mm256_andnot_ps(avx_signbit, to)); // (avx_signbit & from) | (~avx_signbit & to)
}

__attribute__((force_inline))
__m256 _fast_gaussian_cdf(__m256 x) {
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 minusTwoInvPi = _mm256_set1_ps(-2.0f / M_PI);
  
  __m256 xSquared = _mm256_mul_ps(x, x);
  __m256 afterExp = exp256_ps(_mm256_mul_ps(minusTwoInvPi, xSquared));
  __m256 afterSqrt = _mm256_sqrt_ps(_mm256_sub_ps(one, afterExp));
  __m256 afterSqrtSigned = copysign_ps(x, afterSqrt);

  return _mm256_mul_ps(half, _mm256_add_ps(one, afterSqrtSigned));
}

float _fast_gaussian_cdf(float x){
  return 0.5 * (1 + std::copysign(std::sqrt(1 - std::exp(-2 * x * x / M_PI)), x));
}

template<int K> 
std::tuple<float, float> _fast_gmm_cdf(
  float x1, float x2,
  const std::array<float, K> &means, 
  const std::array<float, K> &scales, 
  const std::array<float, K> &weights) {
  float cdf1 = 0.0, cdf2 = 0.0;

  if constexpr (K == 4) { // Kが4の場合のAVX最適化
    __m128 x1Simd = _mm_set1_ps(static_cast<float>(x1));
    __m128 x2Simd = _mm_set1_ps(static_cast<float>(x2));
    __m256 x1x2Simd = _mm256_set_m128(x1Simd, x2Simd); // x2Simd が low, x1Simd が high

    __m128 meansHalf = _mm_loadu_ps(reinterpret_cast<const float*>(means.data()));
    __m256 meansSimd = _mm256_set_m128(meansHalf, meansHalf);
    __m128 scalesHalf = _mm_loadu_ps(reinterpret_cast<const float*>(scales.data()));
    __m256 scalesSimd = _mm256_set_m128(scalesHalf, scalesHalf);
    __m128 weightsHalf = _mm_loadu_ps(reinterpret_cast<const float*>(weights.data()));
    __m256 weightsSimd = _mm256_set_m128(weightsHalf, weightsHalf);

    __m256 x1x2Normalized = _mm256_div_ps(_mm256_sub_ps(x1x2Simd, meansSimd), scalesSimd);
    __m256 cdfs = _mm256_mul_ps(weightsSimd, _fast_gaussian_cdf(x1x2Normalized));

    // cdfs は [c2_0, c2_1, c2_2, c2_3, c1_0, c1_1, c1_2, c1_3] のような順序 (x2がlow 128bit)
    __m128 cdf2Simd_parts = _mm256_castps256_ps128(cdfs);      // low 128 bits
    __m128 cdf1Simd_parts = _mm256_extractf128_ps(cdfs, 1); // high 128 bits
    
    // 水平加算 (hadd)
    cdf1Simd_parts = _mm_hadd_ps(cdf1Simd_parts, cdf1Simd_parts);
    cdf1Simd_parts = _mm_hadd_ps(cdf1Simd_parts, cdf1Simd_parts);
    cdf1 = _mm_cvtss_f32(cdf1Simd_parts);

    cdf2Simd_parts = _mm_hadd_ps(cdf2Simd_parts, cdf2Simd_parts);
    cdf2Simd_parts = _mm_hadd_ps(cdf2Simd_parts, cdf2Simd_parts);
    cdf2 = _mm_cvtss_f32(cdf2Simd_parts);

  } else { // Kが4でない場合の汎用ループ
    for (int i = 0; i < K; ++i){
      cdf1 += weights[i] * _fast_gaussian_cdf((x1 - means[i])/scales[i]);
      cdf2 += weights[i] * _fast_gaussian_cdf((x2 - means[i])/scales[i]);
    }
  }
  return {cdf1, cdf2};
}

/* Support only 16 bits word max */
inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr, uint32_t val,
                             uint32_t nbits) {
  assert(nbits <= 16);
  assert(val < (1u << nbits));

  /* Re-normalize */
  uint64_t x = *r;
  uint32_t freq = 1 << (16 - nbits);
  uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *pptr -= 1;
    **pptr = (uint32_t)x;
    x >>= 32;
    Rans64Assert(x < x_max);
  }

  /* x = C(s, x) */
  *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr,
                                 uint32_t n_bits) {
  uint64_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);

  /* Re-normalize */
  x = x >> n_bits;
  if (x < RANS64_L) {
    x = (x << 32) | **pptr;
    *pptr += 1;
    Rans64Assert(x >= RANS64_L);
  }

  *r = x;

  return val;
}
} // namespace

void BufferedRansEncoder::encode_with_indexes(
    const std::vector<int32_t> &symbols, const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  // backward loop on symbols from the end;
  for (size_t i = 0; i < symbols.size(); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(static_cast<size_t>(cdf_idx) < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert(static_cast<size_t>(max_value + 1) < cdf.size());

    int32_t value = symbols[i] - offsets[cdf_idx];

    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = -2 * value - 1;
      value = max_value;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
      value = max_value;
    }

    assert(value >= 0);
    assert(value < cdfs_sizes[cdf_idx] - 1);

    _syms.push_back({static_cast<uint16_t>(cdf[value]),
                     static_cast<uint16_t>(cdf[value + 1] - cdf[value]),
                     false});

    /* Bypass coding mode (value == max_value -> sentinel flag) */
    if (value == max_value) {
      /* Determine the number of bypasses (in bypass_precision size) needed to
       * encode the raw value. */
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
        ++n_bypass;
      }

      /* Encode number of bypasses */
      int32_t val_bypass = n_bypass; // Renamed to avoid conflict with 'val' from outer scope if any
      while (val_bypass >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val_bypass -= max_bypass_val;
      }
      _syms.push_back(
          {static_cast<uint16_t>(val_bypass), static_cast<uint16_t>(val_bypass + 1), true});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val_bypass_raw =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back(
            {static_cast<uint16_t>(val_bypass_raw), static_cast<uint16_t>(val_bypass_raw + 1), true});
      }
    }
  }
}

void BufferedRansEncoder::encode_with_indexes(
    const std::vector<int32_t> &symbols, const std::vector<float> &scales,
    const int32_t /*max_value*/) { // max_value is unused with gaussian cdf

  // backward loop on symbols from the end;
  for (size_t i = 0; i < symbols.size(); ++i) {

    int32_t value = symbols[i];
    bool bypass = false;
 
    int32_t cdf_value = static_cast<uint16_t>(_fast_gaussian_cdf((value - offset)/scales[i]) * max_cdf_value);
    int32_t cdf_value_next = static_cast<uint16_t>(_fast_gaussian_cdf(((value - offset +1))/scales[i]) * max_cdf_value);

    uint16_t pmf = cdf_value_next - cdf_value;
    if (pmf == 0) { // if pmf is 0, use bypass mode
      bypass = true;
      cdf_value = max_cdf_value; // sentinel value for bypass
      cdf_value_next = max_cdf_value + 1; // so pmf is 1
    }

    _syms.push_back({static_cast<uint16_t>(cdf_value),
                     static_cast<uint16_t>(cdf_value_next - cdf_value), // pmf should be 1 for bypass
                     false}); // bypass flag here is for symbol type, not rans state

    if (bypass) {
      uint32_t raw_val = reinterpret_cast<uint32_t&>(value);
      int32_t n_bypass = 0;
      // Determine number of bypass symbols needed
      // Check up to 32 bits for an int32_t. Max 8 symbols of 4 bits.
      uint32_t temp_raw_val = raw_val;
      while (temp_raw_val != 0 && n_bypass * bypass_precision < 32) {
          temp_raw_val >>= bypass_precision;
          ++n_bypass;
      }
      if (raw_val != 0 && n_bypass == 0) n_bypass = 1; // if raw_val is small but non-zero


      /* Encode number of bypasses */
      int32_t val_nbypass = n_bypass;
      while (val_nbypass >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val_nbypass -= max_bypass_val;
      }
      _syms.push_back(
          {static_cast<uint16_t>(val_nbypass), static_cast<uint16_t>(val_nbypass + 1), true});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val_raw_bits =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back(
            {static_cast<uint16_t>(val_raw_bits), static_cast<uint16_t>(val_raw_bits + 1), true});
      }
    }
  }
}

template<int K>
void BufferedRansEncoder::encode_with_indexes_gmm(
    const torch::Tensor &symbols, const torch::Tensor &scales,
    const torch::Tensor &means, const torch::Tensor &weights,
    const int32_t /*max_value_unused*/) { // max_value is unused with GMM CDF

    // TORCH_CHECK(symbols.is_contiguous() && symbols.scalar_type() == torch::kInt32 && symbols.dim() == 1, 
    //             "symbols tensor must be a contiguous 1D int32 tensor.");
    // TORCH_CHECK(scales.is_contiguous() && scales.scalar_type() == torch::kFloat32 && scales.dim() == 2 && scales.size(1) == K,
    //             "scales tensor must be a contiguous 2D float32 tensor with K columns.");
    // TORCH_CHECK(means.is_contiguous() && means.scalar_type() == torch::kFloat32 && means.dim() == 2 && means.size(1) == K,
    //             "means tensor must be a contiguous 2D float32 tensor with K columns.");
    // TORCH_CHECK(weights.is_contiguous() && weights.scalar_type() == torch::kFloat32 && weights.dim() == 2 && weights.size(1) == K,
    //             "weights tensor must be a contiguous 2D float32 tensor with K columns.");
    // TORCH_CHECK(symbols.size(0) == scales.size(0) && symbols.size(0) == means.size(0) && symbols.size(0) == weights.size(0),
    //             "All input tensors must have the same number of rows (symbols).");
// 
    const int64_t num_symbols = symbols.size(0);
    auto symbols_ptr = symbols.data_ptr<int32_t>();
    auto scales_acc = scales.accessor<float, 2>();
    auto means_acc = means.accessor<float, 2>();
    auto weights_acc = weights.accessor<float, 2>();

    std::array<float, K> current_means_arr;
    std::array<float, K> current_scales_arr;
    std::array<float, K> current_weights_arr;

  // backward loop on symbols from the end;
  for (int64_t i = 0; i < num_symbols; ++i) {
    int32_t value = symbols_ptr[i];
    bool bypass = false;

    for (int k_idx = 0; k_idx < K; ++k_idx) {
        current_means_arr[k_idx] = means_acc[i][k_idx];
        current_scales_arr[k_idx] = scales_acc[i][k_idx];
        current_weights_arr[k_idx] = weights_acc[i][k_idx];
    }
 
    float cdf1, cdf2;
    std::tie(cdf1, cdf2) = _fast_gmm_cdf<K>(
      static_cast<float>(value) - offset, static_cast<float>(value) - offset + 1.0f,
      current_means_arr, current_scales_arr, current_weights_arr
    );

    // Debug output (optional)
    // if (i == 14779) { // Example debug condition
    //   std::cout <<"e (idx " << i <<"): "<< value << " -> cdf1: " << cdf1 << ", cdf2: " << cdf2 << std::endl;
    // }


    int32_t cdf_value = static_cast<uint16_t>(cdf1 * max_cdf_value);
    int32_t cdf_value_next = static_cast<uint16_t>(cdf2 * max_cdf_value);

    uint16_t pmf = cdf_value_next - cdf_value;
    if (pmf == 0) { // if pmf is 0, use bypass mode
      bypass = true;
      cdf_value = max_cdf_value; // sentinel value for bypass
      cdf_value_next = max_cdf_value + 1; // so pmf is 1
    }

    _syms.push_back({static_cast<uint16_t>(cdf_value),
                     static_cast<uint16_t>(cdf_value_next - cdf_value), // pmf should be 1 for bypass
                     false}); // bypass flag here is for symbol type, not rans state


    if (bypass) {
      uint32_t raw_val = reinterpret_cast<uint32_t&>(value);
      int32_t n_bypass = 0;
      // Determine number of bypass symbols needed
      // Check up to 32 bits for an int32_t. Max 8 symbols of 4 bits.
      uint32_t temp_raw_val = raw_val;
      while (temp_raw_val != 0 && n_bypass * bypass_precision < 32) {
          temp_raw_val >>= bypass_precision;
          ++n_bypass;
      }
      if (raw_val != 0 && n_bypass == 0) n_bypass = 1; // if raw_val is small but non-zero

      /* Encode number of bypasses */
      int32_t val_nbypass = n_bypass;
      while (val_nbypass >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val_nbypass -= max_bypass_val;
      }
      _syms.push_back(
          {static_cast<uint16_t>(val_nbypass), static_cast<uint16_t>(val_nbypass + 1), true});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val_raw_bits =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back(
            {static_cast<uint16_t>(val_raw_bits), static_cast<uint16_t>(val_raw_bits + 1), true});
      }
    }
  }
}


py::bytes BufferedRansEncoder::flush() {
  Rans64State rans;
  Rans64EncInit(&rans);

  // std::cout << "Compress Flush" << std::endl;

  std::vector<uint32_t> output_buffer(_syms.size() + 16, 0xCC); // Allocate some extra space
  uint32_t *ptr = output_buffer.data() + output_buffer.size();
  assert(ptr != nullptr);

  // Encode symbols in reverse order (they were pushed_back in forward order of processing,
  // but rANS processes in reverse symbol order)
  std::reverse(_syms.begin(), _syms.end());

  for(const auto& sym : _syms) {
    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    } else {
      Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    }
  }
  _syms.clear(); // Clear symbols after encoding

  Rans64EncFlush(&rans, &ptr);

  const int nbytes =
      std::distance(ptr, output_buffer.data() + output_buffer.size()) * sizeof(uint32_t);
  return std::string(reinterpret_cast<char *>(ptr), nbytes);
}

py::bytes
RansEncoder::encode_with_indexes(const std::vector<int32_t> &symbols,
                                 const std::vector<int32_t> &indexes,
                                 const std::vector<std::vector<int32_t>> &cdfs,
                                 const std::vector<int32_t> &cdfs_sizes,
                                 const std::vector<int32_t> &offsets) {

  BufferedRansEncoder buffered_rans_enc;
  buffered_rans_enc.encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes,
                                        offsets);
  return buffered_rans_enc.flush();
}

py::bytes
RansEncoder::encode_with_indexes(const std::vector<int32_t> &symbols, const std::vector<float> &scales,
    const int32_t max_value) {

  BufferedRansEncoder buffered_rans_enc;
  buffered_rans_enc.encode_with_indexes(symbols, scales, max_value);
  return buffered_rans_enc.flush();
}

template<int K>
py::bytes RansEncoder::encode_with_indexes_gmm(
    const torch::Tensor &symbols, const torch::Tensor &scales,
    const torch::Tensor &means, const torch::Tensor &weights,
    const int32_t max_value) {
  BufferedRansEncoder buffered_rans_enc;
  buffered_rans_enc.encode_with_indexes_gmm<K>(symbols, scales, means, weights, max_value);
  return buffered_rans_enc.flush();
}

std::vector<int32_t>
RansDecoder::decode_with_indexes(const std::string &encoded,
                                 const std::vector<int32_t> &indexes,
                                 const std::vector<std::vector<int32_t>> &cdfs,
                                 const std::vector<int32_t> &cdfs_sizes,
                                 const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  std::vector<int32_t> output(indexes.size());

  Rans64State rans;
  uint32_t *ptr = const_cast<uint32_t *>(reinterpret_cast<const uint32_t *>(encoded.data()));
  assert(ptr != nullptr);
  Rans64DecInit(&rans, &ptr);

  for (size_t i = 0; i < indexes.size(); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(static_cast<size_t>(cdf_idx) < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert(static_cast<size_t>(max_value + 1) < cdf.size());

    const int32_t offset_val = offsets[cdf_idx]; // Renamed to avoid conflict

    const uint32_t cum_freq = Rans64DecGet(&rans, precision);

    const auto cdf_end = cdf.begin() + cdfs_sizes[cdf_idx];
    const auto it = std::lower_bound(cdf.begin(), cdf_end, cum_freq + 1); // find first element > cum_freq
    assert(it != cdf.begin()); // cum_freq must be >= cdf[0] (which is 0)
    const uint32_t s = std::distance(cdf.begin(), it) - 1;


    Rans64DecAdvance(&rans, &ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val_bypass = Rans64DecGetBits(&rans, &ptr, bypass_precision);
      int32_t n_bypass = val_bypass;

      while (val_bypass == max_bypass_val) {
        val_bypass = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        n_bypass += val_bypass;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val_bypass = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        assert(val_bypass <= max_bypass_val);
        raw_val |= val_bypass << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    output[i] = value + offset_val;
  }

  return output;
}

std::vector<int32_t>
RansDecoder::decode_with_indexes(const std::string &encoded,
                                 const std::vector<float> &scales,
                                 const int32_t max_bs_value) {
  std::vector<int32_t> output(scales.size());

  Rans64State rans;
  uint32_t *ptr = const_cast<uint32_t *>(reinterpret_cast<const uint32_t *>(encoded.data()));
  assert(ptr != nullptr);
  Rans64DecInit(&rans, &ptr);

  for (size_t i = 0; i < scales.size(); ++i) {
    float scale_value = scales[i];

    const uint32_t cum_freq = Rans64DecGet(&rans, precision);
    int32_t value;
    if(cum_freq == max_cdf_value) { // Bypass marker
      Rans64DecAdvance(&rans, &ptr, max_cdf_value, 1, precision);
      int32_t val_bypass = Rans64DecGetBits(&rans, &ptr, bypass_precision);
      int32_t n_bypass = val_bypass;

      while (val_bypass == max_bypass_val) {
        val_bypass = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        n_bypass += val_bypass;
      }

      uint32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val_bypass = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        assert(val_bypass <= max_bypass_val);
        raw_val |= val_bypass << (j * bypass_precision);
      }
      value = reinterpret_cast<int32_t&>(raw_val);
    } else {
      int32_t s_bs = -max_bs_value;
      int32_t e_bs = max_bs_value; 
      int32_t mid = 0;
      uint16_t mid_val_1_cdf = 0;
      uint16_t mid_val_2_cdf = 0;

      // Binary search for the symbol
      while (s_bs <= e_bs){ // Use <= for binary search to ensure range is covered
        mid = s_bs + (e_bs - s_bs) / 2;
        mid_val_1_cdf = static_cast<uint16_t>(_fast_gaussian_cdf((static_cast<float>(mid) - offset)/scale_value) * max_cdf_value);
        mid_val_2_cdf = static_cast<uint16_t>(_fast_gaussian_cdf((static_cast<float>(mid) - offset + 1.0f)/scale_value) * max_cdf_value);
        
        if (mid_val_1_cdf <= cum_freq && mid_val_2_cdf > cum_freq) {
            break; // Found the symbol
        } else if (mid_val_1_cdf > cum_freq) { // mid_val_1_cdf is too high, search in lower half
            e_bs = mid - 1;
        } else { // mid_val_2_cdf <= cum_freq, search in upper half
            s_bs = mid + 1;
        }
      }
       // Fallback if exact bracket not found by loop condition (e.g. cum_freq at boundary)
      if (!(mid_val_1_cdf <= cum_freq && mid_val_2_cdf > cum_freq)) {
           mid_val_1_cdf = static_cast<uint16_t>(_fast_gaussian_cdf((static_cast<float>(mid) - offset)/scale_value) * max_cdf_value);
           mid_val_2_cdf = static_cast<uint16_t>(_fast_gaussian_cdf((static_cast<float>(mid) - offset + 1.0f)/scale_value) * max_cdf_value);
      }


      uint16_t pmf = mid_val_2_cdf - mid_val_1_cdf;
      if (pmf == 0 && mid_val_1_cdf <= cum_freq) { // If pmf is 0, means it's likely a very narrow distribution or at tail.
          pmf = 1; // Ensure pmf is at least 1, might need adjustment for exact CDF values
          if (mid_val_1_cdf + pmf > max_cdf_value) mid_val_1_cdf = max_cdf_value -1; // Ensure start+range is valid
      }


      Rans64DecAdvance(&rans, &ptr, mid_val_1_cdf, pmf, precision);
      value = static_cast<int32_t>(mid);
    }
    output[i] = value;
  }
  return output;
}

template<int K> 
torch::Tensor RansDecoder::decode_with_indexes_gmm(const std::string &encoded,
                                 const torch::Tensor &scales,
                                 const torch::Tensor &means,
                                 const torch::Tensor &weights,
                                 const int32_t max_bs_value) {
    // TORCH_CHECK(scales.is_contiguous() && scales.scalar_type() == torch::kFloat32 && scales.dim() == 2 && scales.size(1) == K,
    //             "scales tensor must be a contiguous 2D float32 tensor with K columns.");
    // TORCH_CHECK(means.is_contiguous() && means.scalar_type() == torch::kFloat32 && means.dim() == 2 && means.size(1) == K,
    //             "means tensor must be a contiguous 2D float32 tensor with K columns.");
    // TORCH_CHECK(weights.is_contiguous() && weights.scalar_type() == torch::kFloat32 && weights.dim() == 2 && weights.size(1) == K,
    //             "weights tensor must be a contiguous 2D float32 tensor with K columns.");
    // TORCH_CHECK(scales.size(0) == means.size(0) && scales.size(0) == weights.size(0),
    //             "All GMM parameter tensors must have the same number of rows (symbols).");

  const int64_t num_symbols = scales.size(0);
  auto output = torch::empty({num_symbols}, torch::kInt32);
  auto output_ptr = output.data_ptr<int32_t>();

  auto scales_acc = scales.accessor<float, 2>();
  auto means_acc = means.accessor<float, 2>();
  auto weights_acc = weights.accessor<float, 2>();

  std::array<float, K> current_means_arr;
  std::array<float, K> current_scales_arr;
  std::array<float, K> current_weights_arr;

  Rans64State rans;
  uint32_t *ptr = const_cast<uint32_t *>(reinterpret_cast<const uint32_t *>(encoded.data()));
  assert(ptr != nullptr);
  Rans64DecInit(&rans, &ptr);

  for (int64_t i = 0; i < num_symbols; ++i) {
    for (int k_idx = 0; k_idx < K; ++k_idx) {
        current_means_arr[k_idx] = means_acc[i][k_idx];
        current_scales_arr[k_idx] = scales_acc[i][k_idx];
        current_weights_arr[k_idx] = weights_acc[i][k_idx];
    }

    const uint32_t cum_freq = Rans64DecGet(&rans, precision);
    int32_t value;

    if(cum_freq == max_cdf_value) { // Bypass marker
      Rans64DecAdvance(&rans, &ptr, max_cdf_value, 1, precision);
      int32_t val_bypass = Rans64DecGetBits(&rans, &ptr, bypass_precision);
      int32_t n_bypass = val_bypass;

      while (val_bypass == max_bypass_val) {
        val_bypass = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        n_bypass += val_bypass;
      }

      uint32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val_bypass = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        assert(val_bypass <= max_bypass_val);
        raw_val |= val_bypass << (j * bypass_precision);
      }
      value = reinterpret_cast<int32_t&>(raw_val);
    } else {
      int32_t s_bs = -max_bs_value;
      int32_t e_bs = max_bs_value;
      int32_t mid = 0;
      uint16_t mid_val_1_cdf = 0;
      uint16_t mid_val_2_cdf = 0;
      
      // Binary search for the symbol
      while (s_bs <= e_bs){
        mid = s_bs + (e_bs - s_bs) / 2;
        float cdf1_float, cdf2_float;
        std::tie(cdf1_float, cdf2_float) = _fast_gmm_cdf<K>(
          static_cast<float>(mid) - offset, static_cast<float>(mid) - offset + 1.0f,
          current_means_arr, current_scales_arr, current_weights_arr
        );
        // if (i == 14779) { // Example debug condition
        //    std::cout <<"d (idx " << i <<"): mid=" << mid << " -> cdf1_f: " << cdf1_float << ", cdf2_f: " << cdf2_float << " (target cf: " << cum_freq << ")" << std::endl;
        // }

        mid_val_1_cdf = static_cast<uint16_t>(cdf1_float * max_cdf_value);
        mid_val_2_cdf = static_cast<uint16_t>(cdf2_float * max_cdf_value);
        
        if (mid_val_1_cdf <= cum_freq && mid_val_2_cdf > cum_freq) {
            break; // Found the symbol
        } else if (mid_val_1_cdf > cum_freq) {
            e_bs = mid - 1;
        } else { 
            s_bs = mid + 1;
        }
      }
      // Ensure cdf values correspond to the final 'mid' after loop
      float final_cdf1_float, final_cdf2_float;
      std::tie(final_cdf1_float, final_cdf2_float) = _fast_gmm_cdf<K>(
          static_cast<float>(mid) - offset, static_cast<float>(mid) - offset + 1.0f,
          current_means_arr, current_scales_arr, current_weights_arr
      );
      mid_val_1_cdf = static_cast<uint16_t>(final_cdf1_float * max_cdf_value);
      mid_val_2_cdf = static_cast<uint16_t>(final_cdf2_float * max_cdf_value);


      uint16_t pmf = mid_val_2_cdf - mid_val_1_cdf;
      if (pmf == 0) { // If pmf is 0, ensure it's at least 1 for rANS.
          pmf = 1;
          // Adjust start if necessary to prevent cdf_start + pmf > total_cdf_range
          if (mid_val_1_cdf + pmf > (1 << precision)) { 
              mid_val_1_cdf = (1 << precision) - pmf;
          }
           // This can happen if cum_freq itself implies a symbol with zero probability mass.
           // The binary search might settle on a symbol adjacent to the "true" one.
           // Or if the CDF is extremely steep.
      }

      Rans64DecAdvance(&rans, &ptr, mid_val_1_cdf, pmf, precision);
      value = static_cast<int32_t>(mid);
    }
    output_ptr[i] = value;
  }
  return output;
}


void RansDecoder::set_stream(const std::string &encoded) {
  _stream = encoded;
  uint32_t *ptr = const_cast<uint32_t *>(reinterpret_cast<const uint32_t *>(_stream.data()));
  assert(ptr != nullptr);
  _ptr = ptr;
  Rans64DecInit(&_rans, &_ptr);
}

std::vector<int32_t>
RansDecoder::decode_stream(const std::vector<int32_t> &indexes,
                           const std::vector<std::vector<int32_t>> &cdfs,
                           const std::vector<int32_t> &cdfs_sizes,
                           const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  std::vector<int32_t> output(indexes.size());

  assert(_ptr != nullptr); // Ensure set_stream was called

  for (size_t i = 0; i < indexes.size(); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(static_cast<size_t>(cdf_idx) < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert(static_cast<size_t>(max_value + 1) < cdf.size());

    const int32_t offset_val = offsets[cdf_idx];

    const uint32_t cum_freq = Rans64DecGet(&_rans, precision);

    const auto cdf_end = cdf.begin() + cdfs_sizes[cdf_idx];
    const auto it = std::lower_bound(cdf.begin(), cdf_end, cum_freq + 1);
    assert(it != cdf.begin());
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&_rans, &_ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val_bypass = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
      int32_t n_bypass = val_bypass;

      while (val_bypass == max_bypass_val) {
        val_bypass = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        n_bypass += val_bypass;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val_bypass = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        assert(val_bypass <= max_bypass_val);
        raw_val |= val_bypass << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }
    output[i] = value + offset_val;
  }
  return output;
}

// K_gmm_default is already defined above
// constexpr int K = 4; // Replaced by K_gmm_default for pybind context

PYBIND11_MODULE(ans, m) {
  m.attr("__name__") = "compressai.ans";
  m.doc() = "range Asymmetric Numeral System python bindings";

  py::class_<BufferedRansEncoder>(m, "BufferedRansEncoder")
      .def(py::init<>())
      .def("encode_with_indexes",
           py::overload_cast<
               const std::vector<int32_t> &, const std::vector<int32_t> &,
               const std::vector<std::vector<int32_t>> &,
               const std::vector<int32_t> &, const std::vector<int32_t> &>(
               &BufferedRansEncoder::encode_with_indexes))
      .def("encode_with_indexes",
           py::overload_cast<
               const std::vector<int32_t> &, const std::vector<float> &, const int32_t>(
               &BufferedRansEncoder::encode_with_indexes))
      .def("encode_with_indexes_gmm",
            static_cast<void (BufferedRansEncoder::*) (
                const torch::Tensor &, const torch::Tensor &, // Changed types
                const torch::Tensor &, const torch::Tensor &, // Changed types
                const int32_t)>(
                &BufferedRansEncoder::encode_with_indexes_gmm<K_gmm_default>), // Use K_gmm_default
            py::arg("symbols"), py::arg("scales"), py::arg("means"), py::arg("weights"), py::arg("max_value"))
      .def("flush", &BufferedRansEncoder::flush);

  py::class_<RansEncoder>(m, "RansEncoder")
      .def(py::init<>())
      .def("encode_with_indexes",
           py::overload_cast<
               const std::vector<int32_t> &, const std::vector<int32_t> &,
               const std::vector<std::vector<int32_t>> &,
               const std::vector<int32_t> &, const std::vector<int32_t> &>(
               &RansEncoder::encode_with_indexes))
      .def("encode_with_indexes",
           py::overload_cast<
               const std::vector<int32_t> &, const std::vector<float> &, const int32_t>(
               &RansEncoder::encode_with_indexes))
      .def("encode_with_indexes_gmm",
            static_cast<py::bytes (RansEncoder::*) (
                const torch::Tensor &, const torch::Tensor &, // Changed types
                const torch::Tensor &, const torch::Tensor &, // Changed types
                const int32_t)>(
                &RansEncoder::encode_with_indexes_gmm<K_gmm_default>), // Use K_gmm_default
            py::arg("symbols"), py::arg("scales"), py::arg("means"), py::arg("weights"), py::arg("max_value"));

  py::class_<RansDecoder>(m, "RansDecoder")
      .def(py::init<>())
      .def("set_stream", &RansDecoder::set_stream, py::arg("encoded"))
      .def("decode_stream",
           py::overload_cast<const std::vector<int32_t> &,
                             const std::vector<std::vector<int32_t>> &,
                             const std::vector<int32_t> &,
                             const std::vector<int32_t> &>(
               &RansDecoder::decode_stream))
      .def("decode_with_indexes",
           py::overload_cast<const std::string &, const std::vector<int32_t> &,
                             const std::vector<std::vector<int32_t>> &,
                             const std::vector<int32_t> &,
                             const std::vector<int32_t> &>(
               &RansDecoder::decode_with_indexes),
           "Decode a string to a list of symbols")
      .def("decode_with_indexes",
           py::overload_cast<const std::string &, const std::vector<float> &, const int32_t>(
               &RansDecoder::decode_with_indexes),
           "Decode a string to a list of symbols")
      .def("decode_with_indexes_gmm",
            static_cast<torch::Tensor (RansDecoder::*) ( // Changed return type
                              const std::string &, 
                              const torch::Tensor &, // Changed types
                              const torch::Tensor &, // Changed types
                              const torch::Tensor &, // Changed types
                              const int32_t)>(
                &RansDecoder::decode_with_indexes_gmm<K_gmm_default>), // Use K_gmm_default
            "Decode a string to a tensor of symbols",
            py::arg("encoded"), py::arg("scales"), py::arg("means"), py::arg("weights"), py::arg("max_bs_value"));
}