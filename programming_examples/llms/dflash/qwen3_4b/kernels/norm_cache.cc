// Bit-preserving replay of one chunk from a resident normalized activation.
#include <aie_api/aie.hpp>
#include <cstdint>
// The B16 residual owners emit narrow [chunk128, row8, 128] packets.
// Regroup four packets into [row8, 512] without repeating normalization.
extern "C" void b16_normalized_cache_chunk(bfloat16 *restrict out,
                                           const bfloat16 *restrict slab,
                                           int32_t chunk) {
  for (int sub = 0; sub < 4; ++sub)
    for (int row = 0; row < 8; ++row)
      for (int j = 0; j < 128; j += 64)
        aie::store_v(
            out + row * 512 + sub * 128 + j,
            aie::load_v<64>(slab + (chunk * 4 + sub) * 1024 + row * 128 + j));
}
extern "C" void normalized_cache_chunk(bfloat16 *restrict out,
                                       const bfloat16 *restrict slab,
                                       int32_t chunk) {
  const bfloat16 *src = slab + chunk * 4096;
  for (int i = 0; i < 4096; i += 64) {
    aie::store_v(out + i, aie::load_v<64>(src + i));
  }
}
