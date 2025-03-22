#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include "common.h"
#include "sampling.h"

// A working example of llguidance correctly constraining when not using sampler chain.

int main() {
    const auto model_path = "/home/blackroot/Desktop/llguidance-repro/Llama-3.2-1B-Instruct-IQ4_XS.gguf";

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999;
    llama_model* model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to initialize context" << std::endl;
        llama_model_free(model);
        return 1;
    }

    std::cout << "Model and context loaded successfully" << std::endl;

    const auto grammar_data = R"(%llguidance {}
start: "Hello World")";

    llama_sampler* sampler = llama_sampler_init_llg(llama_model_get_vocab(model), "lark", grammar_data);

    const auto prompt = "Respond hello world.";
    std::cout << "Prompt: " << prompt << std::endl;

    auto tokens = common_tokenize(ctx, prompt, true);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "Initial decode failed" << std::endl;
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    constexpr int max_tokens = 100;
    std::vector<llama_token> output_tokens;
    const llama_token eos_token = llama_vocab_eos(llama_model_get_vocab(model));

    std::cout << "*******OUTPUT*******\n\n";

    for (int i = 0; i < max_tokens; i++) {
        const float* logits = llama_get_logits(ctx);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

        std::vector<llama_token_data> candidates(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates[token_id] = {token_id, logits[token_id], 0.0f};
        }

        llama_token_data_array candidates_p = {
            candidates.data(), candidates.size(), 0, false
        };

        llama_sampler_apply(sampler, &candidates_p);

        bool has_valid_candidates = false;
        for (size_t j = 0; j < candidates_p.size; j++) {
            if (candidates_p.data[j].logit > -INFINITY) {
                has_valid_candidates = true;
                break;
            }
        }

        if (!has_valid_candidates) {
            std::cout << "\nNo valid tokens according to grammar constraints" << std::endl;
            break;
        }

        if (!candidates_p.sorted) {
            std::sort(candidates_p.data, candidates_p.data + candidates_p.size,
                    [](const llama_token_data& a, const llama_token_data& b) {
                        return a.logit > b.logit;
                    });
            candidates_p.sorted = true;
        }

        llama_token new_token = candidates_p.data[0].id;
        output_tokens.push_back(new_token);

        llama_sampler_accept(sampler, new_token);

        std::string token_text = common_token_to_piece(ctx, new_token);
        std::cout << token_text << std::flush;

        if (new_token == eos_token) {
            break;
        }

        batch = llama_batch_get_one(&new_token, 1);

        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "\nDecode failed" << std::endl;
            break;
        }
    }

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}