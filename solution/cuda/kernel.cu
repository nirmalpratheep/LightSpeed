// Naive CUDA-track baseline for FP8 block-scale MoE.
//
// This is a torch C++/CUDA extension (binding = "torch", language = "cuda")
// that mirrors solution/triton/kernel_ref.py exactly. It uses at::Tensor
// torch ops on the GPU rather than hand-written CUDA kernels — the goal here
// is correctness as a reference baseline, not performance. The framework
// JIT-compiles this file with torch.utils.cpp_extension.load via the
// TorchBuilder and calls the symbol named in config.toml entry_point
// ("kernel.cu::kernel").

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int64_t H = 7168;
constexpr int64_t I = 2048;
constexpr int64_t BLOCK = 128;
constexpr int64_t TOP_K = 8;
constexpr int64_t N_GROUP = 8;
constexpr int64_t TOPK_GROUP = 4;

}  // namespace

// Destination-passing-style entry. The framework passes 10 input tensors +
// 1 output tensor (in declaration order from the workload definition):
//
//   routing_logits, routing_bias,
//   hidden_states, hidden_states_scale,
//   gemm1_weights, gemm1_weights_scale,
//   gemm2_weights, gemm2_weights_scale,
//   local_expert_offset, routed_scaling_factor,
//   output
//
// local_expert_offset and routed_scaling_factor arrive as 0-d tensors.
void kernel(
    at::Tensor routing_logits,
    at::Tensor routing_bias,
    at::Tensor hidden_states,
    at::Tensor hidden_states_scale,
    at::Tensor gemm1_weights,
    at::Tensor gemm1_weights_scale,
    at::Tensor gemm2_weights,
    at::Tensor gemm2_weights_scale,
    int64_t local_expert_offset,
    double routed_scaling_factor,
    at::Tensor output) {
  at::NoGradGuard no_grad;

  const int64_t local_start = local_expert_offset;
  const double rsf = routed_scaling_factor;

  const int64_t T = routing_logits.size(0);
  const int64_t E_global = routing_logits.size(1);
  const int64_t E_local = gemm1_weights.size(0);
  const int64_t group_size = E_global / N_GROUP;

  auto device = hidden_states.device();
  auto fp32 = at::TensorOptions().dtype(at::kFloat).device(device);

  // 1) FP8 block-scale dequantization
  auto A_fp32 = hidden_states.to(at::kFloat);
  auto A_scale = hidden_states_scale.to(at::kFloat);          // [H/128, T]
  auto A_scale_TH = A_scale.permute({1, 0}).contiguous();     // [T, H/128]
  auto A_scale_expanded = A_scale_TH.unsqueeze(-1)
                              .repeat({1, 1, BLOCK})
                              .reshape({T, H})
                              .contiguous();
  auto A = A_fp32 * A_scale_expanded;                         // [T, H]

  auto W13_fp32 = gemm1_weights.to(at::kFloat);
  auto S13 = gemm1_weights_scale.to(at::kFloat);
  auto S13_e = at::repeat_interleave(S13, BLOCK, /*dim=*/1);
  S13_e = at::repeat_interleave(S13_e, BLOCK, /*dim=*/2);
  auto W13 = W13_fp32 * S13_e;                                // [E, 2I, H]

  auto W2_fp32 = gemm2_weights.to(at::kFloat);
  auto S2 = gemm2_weights_scale.to(at::kFloat);
  auto S2_e = at::repeat_interleave(S2, BLOCK, /*dim=*/1);
  S2_e = at::repeat_interleave(S2_e, BLOCK, /*dim=*/2);
  auto W2 = W2_fp32 * S2_e;                                   // [E, H, I]

  // 2) No-aux DeepSeek-V3 routing
  auto logits = routing_logits.to(at::kFloat);
  auto bias = routing_bias.to(at::kFloat).reshape({-1});

  auto s = at::sigmoid(logits);
  auto s_with_bias = s + bias;

  auto s_wb_grouped = s_with_bias.view({T, N_GROUP, group_size});
  auto top2 = at::topk(s_wb_grouped, /*k=*/2, /*dim=*/2,
                       /*largest=*/true, /*sorted=*/false);
  auto top2_vals = std::get<0>(top2);
  auto group_scores = top2_vals.sum(/*dim=*/2);

  auto group_top = at::topk(group_scores, /*k=*/TOPK_GROUP, /*dim=*/1,
                            /*largest=*/true, /*sorted=*/false);
  auto group_idx = std::get<1>(group_top);
  auto group_mask = at::zeros_like(group_scores);
  group_mask.scatter_(1, group_idx, 1.0);
  auto score_mask = group_mask.unsqueeze(2)
                        .expand({T, N_GROUP, group_size})
                        .reshape({T, E_global});

  const float neg_inf = std::numeric_limits<float>::lowest();
  auto scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf);
  auto topk_pair = at::topk(scores_pruned, /*k=*/TOP_K, /*dim=*/1,
                            /*largest=*/true, /*sorted=*/false);
  auto topk_idx = std::get<1>(topk_pair);

  auto M = at::zeros_like(s);
  M.scatter_(1, topk_idx, 1.0);
  auto weights = s * M;
  auto weights_sum = weights.sum(/*dim=*/1, /*keepdim=*/true) + 1e-20;
  weights = (weights / weights_sum) * rsf;

  // 3) Per-expert compute and accumulation
  auto out_fp32 = at::zeros({T, H}, fp32);

  for (int64_t le = 0; le < E_local; ++le) {
    const int64_t ge = local_start + le;
    if (ge < 0 || ge >= E_global) continue;

    auto sel_per_token = (topk_idx == ge).any(/*dim=*/1);
    if (!sel_per_token.any().item<bool>()) continue;

    auto token_idx = at::nonzero(sel_per_token).squeeze(1);

    auto A_e = A.index_select(0, token_idx);          // [Tk, H]
    auto W13_e = W13.select(0, le);                   // [2I, H]
    auto W2_e = W2.select(0, le);                     // [H, I]

    auto G1 = at::matmul(A_e, W13_e.t());             // [Tk, 2I]
    auto X1 = G1.slice(1, 0, I);
    auto X2 = G1.slice(1, I, 2 * I);
    auto silu_X2 = X2 * at::sigmoid(X2);
    auto C = silu_X2 * X1;                            // [Tk, I]

    auto O = at::matmul(C, W2_e.t());                 // [Tk, H]

    auto w_tok = weights.index_select(0, token_idx).select(1, ge);
    out_fp32.index_add_(0, token_idx, O * w_tok.unsqueeze(1));
  }

  output.copy_(out_fp32.to(at::kBFloat16));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kernel", &kernel, "Naive MoE FP8 block-scale baseline (CUDA track)");
}
