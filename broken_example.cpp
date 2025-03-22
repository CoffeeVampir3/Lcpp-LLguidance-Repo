#include <iostream>
#include <vector>
#include <string>
#include "common.h"
#include "sampling.h"

// This is using sampler chain instead of manual sampling.

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

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler* sampler = llama_sampler_chain_init(sparams);

    const auto grammar_data = R"(%llguidance {}
start: "Hello World")";

    llama_sampler_chain_add(sampler, llama_sampler_init_llg(llama_model_get_vocab(model), "lark", grammar_data));

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
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

        std::cout << new_token << std::endl;
        llama_sampler_accept(sampler, new_token);
        output_tokens.push_back(new_token);

        if (llama_vocab_is_eog(llama_model_get_vocab(model), new_token)) {
            break;
        }

        std::string token_text = common_token_to_piece(ctx, new_token);
        std::cout << token_text << std::flush;

        if (new_token == eos_token) {
            break;
        }

        batch = llama_batch_get_one(&new_token, 1);
        std::cout << new_token << std::endl;

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