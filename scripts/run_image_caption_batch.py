import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import json
from pathlib import Path
import time
try:
    from scripts.prepare_captions import rows, dump
except ModuleNotFoundError:
    from prepare_captions import rows, dump

def cumulative_cost(responses):
    return {'input_tokens': sum((r.get('input_tokens', 0) for r in responses)), 'output_tokens': sum((r.get('output_tokens', 0) for r in responses)), 'model_record_calls': sum((r['model_called'] for r in responses)), 'batch_generate_calls': round(sum((1 / r['batch_records'] for r in responses if r['model_called']))), 'cumulative_batch_seconds': sum((r['batch_seconds'] / r['batch_records'] for r in responses))}

def run(args):
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    requests = rows(args.requests)
    if len({r['id'] for r in requests}) != len(requests):
        raise ValueError('Duplicate request IDs')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'model': str(args.model), 'family': args.family, 'max_new_tokens': args.max_new_tokens, 'batch_size': args.batch_size, 'max_patches': args.max_patches, 'do_sample': False, 'attention': 'sdpa', 'dtype': 'bfloat16'}
    dump(args.output / 'config.json', config)
    (args.output / 'requests.jsonl').write_text(args.requests.read_text())
    prior = []
    pending = requests
    processor_args = {'min_pixels': 256 * 28 ** 2, 'max_pixels': 1024 * 28 ** 2} if args.family == 'qwen' else {}
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, **processor_args)
    processor.tokenizer.padding_side = 'left'
    started = time.monotonic()
    model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
    torch.cuda.reset_peak_memory_stats()
    load_seconds = time.monotonic() - started
    receipt = {'config': config, 'requested': len(requests), 'torch': torch.__version__, 'transformers': transformers.__version__, 'load_seconds': load_seconds, 'gpu': torch.cuda.get_device_name(0), 'input_tokens': 0, 'output_tokens': 0, 'batch_generate_calls': 0, 'image_only': True}
    generated = []
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        t0 = time.monotonic()
        inputs = output = None
        result = [{'id': r['id'], 'status': 'runtime_fail', 'response': None, 'model_called': False} for r in batch]
        try:
            processor_kwargs = {'padding': True}
            if args.family == 'internvl':
                processor_kwargs['images_kwargs'] = {'max_patches': args.max_patches}
            inputs = processor.apply_chat_template([r['messages'] for r in batch], processor_kwargs=processor_kwargs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to('cuda:0', dtype=torch.bfloat16)
            width = inputs['input_ids'].shape[1]
            for (i, r) in enumerate(result):
                r.update(input_tokens=int(inputs['attention_mask'][i].sum()), model_called=True)
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            torch.cuda.synchronize()
            for (i, r) in enumerate(result):
                seq = output[i, width:].tolist()
                eos = model.generation_config.eos_token_id
                eos = {eos} if isinstance(eos, int) else set(eos or [])
                n = next((j + 1 for (j, t) in enumerate(seq) if t in eos), len(seq))
                r.update(response=processor.decode(seq[:n], skip_special_tokens=True), output_tokens=n, status='generated' if n < args.max_new_tokens or seq[n - 1] in eos else 'token_limit')
        except Exception as exc:
            for r in result:
                r['error'] = type(exc).__name__ + ': ' + str(exc)
        finally:
            del inputs, output
        elapsed = time.monotonic() - t0
        for r in result:
            r.update(batch_seconds=elapsed, batch_records=len(batch))
        with (args.output / 'responses.jsonl').open('a') as f:
            for r in result:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        generated.extend(result)
        receipt.update(completed=len(prior) + len(generated), status_counts=dict(Counter((r['status'] for r in prior + generated))), wall_seconds=time.monotonic() - started, peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024 ** 3, **cumulative_cost(prior + generated))
        print(json.dumps({'completed': receipt['completed'], 'requested': len(requests), 'batch_seconds': elapsed, 'statuses': dict(Counter((r['status'] for r in result)))}), flush=True)
        if all((r['status'] == 'runtime_fail' for r in result)):
            raise RuntimeError('Whole batch failed; inspect raw outcomes before continuing')
    dump(args.output / 'receipt.json', receipt)
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--family', choices=('internvl', 'qwen'), default='internvl')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=96)
    p.add_argument('--max-patches', type=int, default=1)
    run(p.parse_args())
