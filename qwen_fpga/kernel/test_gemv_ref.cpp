// test_gemv_ref.cpp -- verify the C++ INT4 GEMV datapath against Python golden.
//
// Reads <golden_dir>/cases.tsv and, for each case, loads the packed weights,
// fp16 scales, INT8 activation and expected fp32 output produced by
// qwen_fpga/reference/make_golden.py, runs gemv_int4_compute(), and reports the
// max abs/rel error. Exit non-zero if any case exceeds the tolerance.
//
// This is the bit-level proof that the kernel math matches the quantized Qwen
// reference *before* any HLS synthesis -- Milestone A, extended to every shape.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>

#include "gemv_int4.hpp"

static std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(n);
    f.read(reinterpret_cast<char*>(buf.data()), n);
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <golden_dir> [tol] [max_cases]\n", argv[0]);
        return 2;
    }
    std::string dir = argv[1];
    double tol = (argc > 2) ? atof(argv[2]) : 1e-3;
    int max_cases = (argc > 3) ? atoi(argv[3]) : 0;  // 0 = all

    std::ifstream tsv(dir + "/cases.tsv");
    if (!tsv) { fprintf(stderr, "no cases.tsv in %s\n", dir.c_str()); return 2; }

    std::string line;
    int n_cases = 0, n_fail = 0;
    double worst = 0.0;
    std::string worst_name;
    while (std::getline(tsv, line)) {
        if (line.empty() || line[0] == '#') continue;
        char name[256]; int M, N, gs, ng; double xscale;
        if (sscanf(line.c_str(), "%255[^\t]\t%d\t%d\t%d\t%d\t%lf",
                   name, &M, &N, &gs, &ng, &xscale) != 6) continue;

        std::string base = dir + "/" + name;
        auto W  = read_bytes(base + ".W.int4.bin");
        auto S  = read_bytes(base + ".W.scale.bin");
        auto X  = read_bytes(base + ".x.int8.bin");
        auto Yg = read_bytes(base + ".y.f32.bin");

        const uint16_t* scales = reinterpret_cast<const uint16_t*>(S.data());
        const int8_t* x = reinterpret_cast<const int8_t*>(X.data());
        const float* ygold = reinterpret_cast<const float*>(Yg.data());

        std::vector<float> y(M);
        gemv_int4_compute(W.data(), scales, x, (float)xscale, y.data(), M, N, gs);

        double max_abs = 0.0, ref_max = 1e-12;
        for (int m = 0; m < M; ++m) {
            max_abs = std::max(max_abs, (double)std::fabs(y[m] - ygold[m]));
            ref_max = std::max(ref_max, (double)std::fabs(ygold[m]));
        }
        double rel = max_abs / ref_max;
        if (rel > worst) { worst = rel; worst_name = name; }
        bool ok = rel <= tol;
        if (!ok) {
            ++n_fail;
            printf("  FAIL %-32s M=%-5d N=%-5d max_abs=%.3e rel=%.3e\n",
                   name, M, N, max_abs, rel);
        }
        if (++n_cases <= 8 || !ok)
            printf("  %-4s %-30s M=%-5d N=%-5d rel=%.3e\n",
                   ok ? "ok" : "BAD", name, M, N, rel);
        if (max_cases && n_cases >= max_cases) break;
    }
    printf("\n%d cases, %d failures, worst rel=%.3e (%s), tol=%.1e\n",
           n_cases, n_fail, worst, worst_name.c_str(), tol);
    return n_fail ? 1 : 0;
}
