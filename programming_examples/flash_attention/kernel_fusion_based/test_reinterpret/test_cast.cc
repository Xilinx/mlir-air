// test_cast.cc — Verify reinterpret_cast<float*>(bfloat16*) works on AIE2P.
//
// Tests that we can use a bf16 L1 buffer as f32 storage by pointer casting.
// The matmul accumulator writes f32 to the buffer, a convert function reads
// f32 and writes bf16 back.
//
// Compile:
//   $PEANO_INSTALL_DIR/bin/clang++ -O2 -std=c++20 \
//     --target=aie2p-none-unknown-elf \
//     -I $AIEOPT_DIR/include \
//     -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 \
//     -c test_cast.cc -o test_cast.o

#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>

// Test 1: Write f32 values into a bf16 buffer via reinterpret_cast.
// Buffer is declared as bfloat16* (8192 bf16 elements = 16384 bytes),
// but we treat it as float* (4096 float elements = 16384 bytes).
extern "C" {

void test_write_f32_to_bf16_buf(bfloat16 *__restrict buf, int n_floats) {
    float *__restrict fbuf = reinterpret_cast<float *>(buf);
    for (int i = 0; i < n_floats; i += 32) {
        // Write a known f32 pattern: 1.0, 2.0, 3.0, ...
        aie::vector<float, 32> v;
        for (int j = 0; j < 32; ++j) {
            v[j] = static_cast<float>(i + j + 1);
        }
        aie::store_v(fbuf + i, v);
    }
}

// Test 2: Read f32 from bf16 buffer, convert to bf16, write back.
// After this, the buffer contains bf16 values in the first n_floats*2 bytes
// (since bf16 is 2 bytes vs float's 4 bytes, we write half the buffer).
void test_f32_to_bf16_inplace(bfloat16 *__restrict buf, int n_floats) {
    float *__restrict fbuf = reinterpret_cast<float *>(buf);
    // Read f32 values, convert to bf16, write to beginning of buffer
    for (int i = 0; i < n_floats; i += 32) {
        aie::vector<float, 32> fv = aie::load_v<32>(fbuf + i);
        // Convert float -> accfloat -> bfloat16
        aie::accum<accfloat, 32> acc(fv);
        aie::vector<bfloat16, 32> bv = acc.to_vector<bfloat16>();
        aie::store_v(buf + i, bv);
    }
}

// Test 3: Matmul-style pattern — accumulate in f32, store as f32.
// This mimics what matmul_a_b_bf16 would do if it wrote f32 output.
void test_matmul_f32out(bfloat16 *__restrict pA,
                        bfloat16 *__restrict pB,
                        bfloat16 *__restrict pC_bf16_buf) {
    // Reinterpret the bf16 output buffer as float
    float *__restrict pC = reinterpret_cast<float *>(pC_bf16_buf);

    using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;
    constexpr int rowA = 8;  // lqp/r = 64/8
    constexpr int colA = 8;  // dk/s = 64/8
    constexpr int colB = 8;  // lkp/t = 64/8

    for (unsigned z = 0; z < rowA; z += 2) {
        float *__restrict pC1 = pC + z * MMUL::size_C;
        float *__restrict pC2 = pC + (z + 1) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2) {
            const bfloat16 *__restrict pA1 = pA + z * MMUL::size_A;
            const bfloat16 *__restrict pA2 = pA + (z + 1) * MMUL::size_A;

            // Load existing f32 accumulator from buffer
            aie::vector<float, MMUL::size_C> acc_f00 =
                aie::load_v<MMUL::size_C>(pC1);
            aie::vector<float, MMUL::size_C> acc_f01 =
                aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA);

            // Promote to accfloat for mmul (use brace init to avoid vexing parse)
            aie::accum<accfloat, MMUL::size_C> a00{acc_f00};
            aie::accum<accfloat, MMUL::size_C> a01{acc_f01};
            MMUL C00{a00};
            MMUL C01{a01};

            // ... (mac operations would go here) ...

            // Store back as f32 (NOT bf16 — no truncation!)
            aie::store_v(pC1,
                         C00.template to_vector<float>());
            pC1 += MMUL::size_C * rowA;
            aie::store_v(pC1,
                         C01.template to_vector<float>());
            pC1 += MMUL::size_C * rowA;
        }
    }
}

} // extern "C"
