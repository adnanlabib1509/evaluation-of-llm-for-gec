# Inference Scripts

Scripts for running inference with baseline and fine-tuned models, and formatting results.

## Files

### A03_gpt_inference.py
Runs inference with fine-tuned GPT-4o model using `ModelRunner`. Processes test dataset and saves results.

### A04_gpt_format_output.py
Converts inference results from JSONL to Excel format for analysis. Works with any model (GPT-4o, DeepSeek, etc.).

### C01_inference_parrallel.py
Generic parallel inference script using LiteLLM. Supports any model provider with concurrent requests and rate limiting.

### D01_prepare_openai_batch.py
Prepares test dataset in OpenAI Batch API format for cost-effective bulk inference.

### D02_run_openai_batch.py
Uploads and starts OpenAI batch jobs. Alternative to real-time API calls for large-scale inference.

### E02_llama_inference.py
Runs inference with fine-tuned Llama models. Outputs in same JSONL format as other scripts.
