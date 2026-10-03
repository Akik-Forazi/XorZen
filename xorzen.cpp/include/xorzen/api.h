#pragma once

#ifdef _WIN32
#  ifdef XORZEN_BUILD_DLL
#    define XORZEN_API __declspec(dllexport)
#  else
#    define XORZEN_API __declspec(dllimport)
#  endif
#else
#  define XORZEN_API
#endif

extern "C" {

XORZEN_API void* xorzen_load_model(const char* checkpoint_path);
XORZEN_API int xorzen_forward(void* model, const int* input_ids, int batch_size, int seq_len, float* logits);
XORZEN_API int xorzen_train_step(void* model, const int* input_ids, const int* labels,
                                 int batch_size, int seq_len, float* loss_out);
XORZEN_API int xorzen_generate(void* model, const int* prompt, int prompt_len,
                               int max_new_tokens, int* output, int* output_len);
XORZEN_API int xorzen_save_checkpoint(void* model, const char* path);
XORZEN_API int xorzen_get_metrics(void* model, float* metrics, int max_metrics);
XORZEN_API void xorzen_free_model(void* model);

XORZEN_API void* xorzen_load_tokenizer(const char* tokenizer_json_path);
XORZEN_API int xorzen_tokenizer_encode(void* tokenizer, const char* text,
                                       int* output_ids, int max_tokens,
                                       int add_special_tokens);
XORZEN_API int xorzen_tokenizer_decode(void* tokenizer, const int* token_ids,
                                       int token_count, char* output_text,
                                       int max_chars, int skip_special_tokens);
XORZEN_API void xorzen_free_tokenizer(void* tokenizer);
XORZEN_API int xorzen_rmsnorm(const float* x, const float* weight, int batch, int seq, int dim, float* output);

}
