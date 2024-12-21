/* Copyright (c) 2021-2024, InterDigital Communications, Inc
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of InterDigital Communications, Inc nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
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
// #include <torch/extension.h>

#include <iostream>
#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans64.h"

namespace py = pybind11;

/* probability range, this could be a parameter... */
constexpr int precision = 16;
constexpr int32_t max_cdf_value = 65535;
constexpr float offset = 0.5;
constexpr uint16_t bypass_precision = 4; /* number of bits in bypass mode */
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

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

float _fast_gaussian_cdf(float x){
  return 0.5 * (1 + sgn(x) * std::sqrt(1 - std::exp(-2 * x * x / M_PI)));
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
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

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
      int32_t val = n_bypass;
      while (val >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val -= max_bypass_val;
      }
      _syms.push_back(
          {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back(
            {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});
      }
    }
  }
}

void BufferedRansEncoder::encode_with_indexes(
    const std::vector<int32_t> &symbols, const std::vector<float> &scales,
    const int32_t max_value) {;

  // backward loop on symbols from the end;
  for (size_t i = 0; i < symbols.size(); ++i) {

    int32_t value = symbols[i];
    //std::cout << value << " ";
    bool bypass = false;

    //float value_half = value - offset;
 
    int32_t cdf_value = static_cast<uint16_t>(_fast_gaussian_cdf((value - offset)/scales[i]) * max_cdf_value);
    int32_t cdf_value_next = static_cast<uint16_t>(_fast_gaussian_cdf(((value - offset +1))/scales[i]) * max_cdf_value);

    uint16_t pmf = cdf_value_next - cdf_value;
    if (pmf == 0) {
      bypass = true;
      cdf_value = max_cdf_value;
      cdf_value_next = max_cdf_value + 1;
    }

    _syms.push_back({static_cast<uint16_t>(cdf_value),
                     static_cast<uint16_t>(cdf_value_next - cdf_value),
                     false});

    if (bypass) {
      uint32_t raw_val = reinterpret_cast<uint32_t&>(value);
      /* Bypass coding mode (cdf == max_cdf_value -> sentinel flag) */
      /* Determine the number of bypasses (in bypass_precision size) needed to
       * encode the raw value. */
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
        ++n_bypass;
        if (n_bypass > 8) break;
      } 

      /* Encode number of bypasses */
      int32_t val = n_bypass;
      while (val >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val -= max_bypass_val;
      }
      _syms.push_back(
          {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back(
            {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});
      }
    }
  }
}

py::bytes BufferedRansEncoder::flush() {
  Rans64State rans;
  Rans64EncInit(&rans);

  //std::cout << "Compress Flush" << std::endl;

  std::vector<uint32_t> output(_syms.size(), 0xCC); // too much space ?
  uint32_t *ptr = output.data() + output.size();
  assert(ptr != nullptr);

  while (!_syms.empty()) {
    const RansSymbol sym = _syms.back();

    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    } else {
      // unlikely...
      Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    }
    _syms.pop_back();
  }

  Rans64EncFlush(&rans, &ptr);

  const int nbytes =
      std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
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
  uint32_t *ptr = (uint32_t *)encoded.data();
  assert(ptr != nullptr);
  Rans64DecInit(&rans, &ptr);

  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets[cdf_idx];

    const uint32_t cum_freq = Rans64DecGet(&rans, precision);

    const auto cdf_end = cdf.begin() + cdfs_sizes[cdf_idx];
    const auto it = std::find_if(cdf.begin(), cdf_end,
                                 [cum_freq](int v) { return v > cum_freq; });
    assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&rans, &ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        assert(val <= max_bypass_val);
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    output[i] = value + offset;
  }

  return output;
}

std::vector<int32_t>
RansDecoder::decode_with_indexes(const std::string &encoded,
                                 const std::vector<float> &scales,
                                 const int32_t max_bs_value) {
  std::vector<int32_t> output(scales.size());

  Rans64State rans;
  uint32_t *ptr = (uint32_t *)encoded.data();
  assert(ptr != nullptr);
  Rans64DecInit(&rans, &ptr);

  for (int i = 0; i < static_cast<int>(scales.size()); ++i) {
    float scale_value = scales[i];

    const uint32_t cum_freq = Rans64DecGet(&rans, precision);
    int32_t value;
    if(cum_freq == max_cdf_value) {
      Rans64DecAdvance(&rans, &ptr, max_cdf_value, 1, precision);
            /* Bypass decoding mode */
      int32_t val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        n_bypass += val;
      }

      uint32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        assert(val <= max_bypass_val);
        raw_val |= val << (j * bypass_precision);
      }
      value = reinterpret_cast<int32_t&>(raw_val);
    } else {
      //Search symbol s such that _fast_gaussian_cdf(s/scales) < <= cum_freq
      // and _fast_gaussian_cdf((s+1)/scales) > cum_freq using binary search
      int32_t s = -max_bs_value;
      int32_t e = max_bs_value; //some large value
      int32_t mid = 0;
      int32_t mid_val_1 = 0.0;
      int32_t mid_val_2 = 0.0;
      while (s < e){
        mid = s + (e - s) / 2;
        //float mid_half = mid - offset;
        mid_val_1 = _fast_gaussian_cdf((mid - offset)/scale_value) * max_cdf_value;
        bool check1 = mid_val_1 <= cum_freq;
        mid_val_2 = _fast_gaussian_cdf((mid - offset +1)/scale_value) * max_cdf_value;
        bool check2 = mid_val_2 > cum_freq;
        if(check1 && check2){
          break;
        } else if (check1){
          s = mid + 1;
        } else {
          e = mid; 
        }
      }
      int32_t mid1_int = static_cast<int32_t>(mid_val_1);
      int32_t mid2_int = static_cast<int32_t>(mid_val_2);
      Rans64DecAdvance(&rans, &ptr, mid1_int, mid2_int - mid1_int, precision);

      value = static_cast<int32_t>(mid);
    }

    //std::cout << value << " ";
    output[i] = value;
  }
  //std::cout << "Decode End";
  //std::cout << std::endl;

  return output;
}


void RansDecoder::set_stream(const std::string &encoded) {
  _stream = encoded;
  uint32_t *ptr = (uint32_t *)_stream.data();
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

  assert(_ptr != nullptr);

  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets[cdf_idx];

    const uint32_t cum_freq = Rans64DecGet(&_rans, precision);

    const auto cdf_end = cdf.begin() + cdfs_sizes[cdf_idx];
    const auto it = std::find_if(cdf.begin(), cdf_end,
                                 [cum_freq](int v) { return v > cum_freq; });
    assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&_rans, &_ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        assert(val <= max_bypass_val);
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    output[i] = value + offset;
  }

  return output;
}


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
               &RansEncoder::encode_with_indexes));

  py::class_<RansDecoder>(m, "RansDecoder")
      .def(py::init<>())
      .def("set_stream", &RansDecoder::set_stream)
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
           "Decode a string to a list of symbols");
}
