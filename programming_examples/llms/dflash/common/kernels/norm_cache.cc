// Bit-preserving replay of one chunk from a resident normalized activation.
#include <aie_api/aie.hpp>
#include <cstdint>
#ifndef NORM_CACHE_ROWS
#define NORM_CACHE_ROWS 8
#endif
static_assert(NORM_CACHE_ROWS == 4 || NORM_CACHE_ROWS == 8);
// Regroup two [row8,128] normalized chunks into [row8,256].
extern "C" void b16_normalized_cache_chunk(bfloat16 *restrict out,
                                           const bfloat16 *restrict slab,
                                           int32_t chunk) {
  for (int sub = 0; sub < 2; ++sub)
    for (int row = 0; row < 8; ++row)
      for (int j = 0; j < 128; j += 64)
        aie::store_v(
            out + row * 256 + sub * 128 + j,
            aie::load_v<64>(slab + (chunk * 2 + sub) * 1024 + row * 128 + j));
}
extern "C" void normalized_cache_chunk(bfloat16 *restrict out,
                                       const bfloat16 *restrict slab,
                                       int32_t chunk) {
  const bfloat16 *src = slab + chunk * (NORM_CACHE_ROWS * 512);
  for (int i = 0; i < NORM_CACHE_ROWS * 512; i += 64) {
    aie::store_v(out + i, aie::load_v<64>(src + i));
  }
}

// Separate input/output buffers keep the peer DMA from overwriting an
// outgoing packet while the packet transmitter is still reading it.
extern "C" void b16_normalized_cache_join(bfloat16 *restrict out,
                                          const bfloat16 *restrict peer,
                                          const bfloat16 *restrict slab,
                                          int32_t chunk) {
  for (int i = 0; i < 2048; i += 64)
    aie::store_v(out + i, aie::load_v<64>(peer + i));
  b16_normalized_cache_chunk(out + 2048, slab, chunk);
}
