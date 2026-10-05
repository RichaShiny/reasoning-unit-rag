"""Pinned local encoder-decoder generation using the shared cache contract."""


def make_local_provider(model, revision, device='cpu', max_input_tokens=512, num_beams=1):
    if not revision or max_input_tokens < 1 or num_beams < 1:
        raise ValueError('Revision and positive input/beam limits are required')
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    network = AutoModelForSeq2SeqLM.from_pretrained(model, revision=revision,
                                                   trust_remote_code=False, use_safetensors=True)
    network.to(device)
    network.eval()

    def provider(request):
        prompt = request['instructions'] + '\nQuestion:\n' + request['input'] + '\nSubqueries JSON:\n'
        inputs = tokenizer(prompt, return_tensors='pt', truncation=False)
        input_tokens = inputs['input_ids'].shape[-1]
        if input_tokens > max_input_tokens:
            # Preserve the whole question; never silently cut off its final entities.
            raise ValueError('Local prompt exceeds configured input-token limit')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = network.generate(**inputs, max_new_tokens=request['max_output_tokens'],
                                      do_sample=False, num_beams=num_beams)
        eos = tokenizer.eos_token_id
        completed = eos is not None and int(output[0][-1]) == eos
        return {'raw_text': tokenizer.decode(output[0], skip_special_tokens=True),
                'usage': {'input_tokens': input_tokens,
                          'output_tokens': max(0, output.shape[-1] - 1)},
                'returned_model': f'{model}@{revision}', 'response_id': None,
                'response_status': 'completed' if completed else 'incomplete'}

    return provider
