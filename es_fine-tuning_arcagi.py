import argparse
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

from utils import grid_accuracy, load_arc_agi_dataset, parse_grid_from_text

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--project_path', type=str, default='your/project/path')
parser.add_argument('--hf_cache_dir', type=str, default='hf_cache')
parser.add_argument('--mixed_precision', type=str, default='bf16')
parser.add_argument('--gpu_threads', type=int, default=4, help='Maximum number of parallel workers per GPU')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
parser.add_argument('--data_sample', type=int, default=256, help='Number of data samples to use for training')
parser.add_argument('--seed_concurrency', type=int, default=4)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--max_new_tokens', type=int, default=1024)
parser.add_argument('--log_sample_every', type=int, default=10, help='Log a sample prompt/output every N iterations (0 to disable)')
args = parser.parse_args()


# Hyperparameters for ES
NUM_ITERATIONS = 1000             # Number of ES iterations (generations)
POPULATION_SIZE = 32              # Population size (number of perturbations per iteration)
SIGMA = 0.001                     # Standard deviation for weight perturbations (noise scale)
ALPHA = 0.0005                    # Learning rate
do_sample = False                 # Whether sampling is allowed in generating tokens
initial_seed = 33                 # Initial random seed


def force_memory_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def save_model_checkpoint(model, tokenizer, iteration, model_name, initial_seed, args, dataset_size, *, tag: str | None = None) -> None:
    """Save model checkpoint at specified iteration."""
    question_num = dataset_size
    suffix = tag if tag else f"iter{iteration}"
    save_dir = f"{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_{suffix}_sigma{SIGMA}_alpha{ALPHA}_{args.mixed_precision}_threads{args.gpu_threads}_question_num{question_num}_checkpoint"
    print(f"Saving checkpoint ({suffix}) to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Checkpoint saved successfully.")






def evaluate_model(model, tokenizer, prompt_texts, target_grids, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False):
    """Generate responses for prompts and compute grid-based rewards."""
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    prompts = prompt_texts if isinstance(prompt_texts, list) else [prompt_texts]
    targets = target_grids if isinstance(target_grids, list) else [target_grids]

    tokenized_inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
    )
    input_ids = tokenized_inputs['input_ids'].to(accelerator.device)
    attention_mask = tokenized_inputs.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(accelerator.device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    generated_ids = outputs[:, input_ids.shape[-1]:]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    rewards = []
    for text, target in zip(generated_texts, targets):
        expected_shape = None
        if target and len(target[0]) > 0:
            expected_shape = (len(target), len(target[0]))
        pred_grid = parse_grid_from_text(text, expected_shape=expected_shape)
        rewards.append(grid_accuracy(pred_grid, target))

    del input_ids, outputs
    if attention_mask is not None:
        del attention_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (rewards, generated_texts) if return_text else rewards

def evaluate_model_batched(model, tokenizer, prompts, targets, accelerator, batch_size):
    all_rewards = []
    for i in range(0, len(prompts), batch_size):
        chunk_p = prompts[i:i+batch_size]
        chunk_t = targets[i:i+batch_size]
        r = evaluate_model(
                            model,
                            tokenizer,
                            chunk_p,
                            chunk_t,
                            accelerator,
                            return_text=False,
                        )
        all_rewards.extend(r)
    return all_rewards



def process_seed(seed_args):
    """Function to process a single seed, used for thread pool."""
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose, dataset = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")

    # Weight perturbation
    for _, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype,
        )
        param.data.add_(SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    input_texts = [data['prompt'] for data in dataset]
    target_grids = [data['target'] for data in dataset]
    rewards = evaluate_model_batched(model, tokenizer, input_texts, target_grids, accelerator, args.eval_batch_size) 
    total_reward = sum(rewards)

    for _, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype,
        )
        param.data.add_(-SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    average_reward = total_reward / len(dataset)

    force_memory_cleanup()

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {average_reward:.4f}")

    return seed_idx, average_reward


def main():
    accelerator = Accelerator()
    # # new
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(accelerator.local_process_index)
    #     dev = torch.cuda.current_device()
    #     print(f"[rank {accelerator.process_index}] pinned to cuda:{dev} (local {accelerator.local_process_index})")
    #     assert dev == accelerator.local_process_index

    project_root = args.project_path
    data_root = project_root + '/data/arc-prize-2025'
    challenges_path = data_root + '/arc-agi_training_challenges.json'
    solutions_path = data_root + '/arc-agi_training_solutions.json'

    limit = args.data_sample if args.data_sample and args.data_sample > 0 else None
    dataset_samples = load_arc_agi_dataset(challenges_path, solutions_path, limit=limit)
    if not dataset_samples:
        raise ValueError('No ARC-AGI samples were loaded.')

    if accelerator.is_main_process:
        print(f"Loaded {len(dataset_samples)} ARC-AGI test cases from {challenges_path}")
        print(f"Total processes: {accelerator.num_processes}, GPU threads per process: {args.gpu_threads}")
        print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
        print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")
        print(accelerator.state)

    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir

    if accelerator.is_main_process:
        print(f"Loading model {model_name}...")

    model_list = []
    for model_index in range(args.gpu_threads):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            device_map={"": accelerator.process_index},  # Assign devices explicitly
            torch_dtype=torch.float16 if args.mixed_precision == 'fp16' else (torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float32),
        )
        # # new
        # model.to(accelerator.device)
        # model.eval()
        
        model_list.append(model)
    
    # # new
    # for i, m in enumerate(model_list):
    #     pdev = next(m.parameters()).device
    #     print(f"[rank {accelerator.process_index}] model[{i}] on {pdev}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if accelerator.is_main_process: print('Model loaded successfully')

    dataset = []
    for sample in dataset_samples:
        prompt_text = sample.build_prompt(tokenizer)
        dataset.append(
            {
                'task_id': sample.task_id,
                'prompt': prompt_text,
                'target': sample.target,
            }
        )
    dataset_size = len(dataset)
    if accelerator.is_main_process and dataset:
        print('Dataset sample:', dataset[0])

    accelerator.wait_for_everyone()

    force_memory_cleanup()
    training_start_time = time.time()

    np.random.seed(initial_seed)

    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()

        force_memory_cleanup()

        if args.verbose:
            print(f"Process {accelerator.process_index} starting iteration {iteration + 1}/{NUM_ITERATIONS}")

        if accelerator.is_main_process:
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)

        if accelerator.num_processes > 1:
            torch.distributed.broadcast(seeds_tensor, src=0)
            # # new
            # print(f"[rank {accelerator.process_index}] about to broadcast on device {torch.cuda.current_device()}")
            
        seeds = seeds_tensor.cpu().tolist()

        local_seeds = [
            (seed_idx, seed)
            for seed_idx, seed in enumerate(seeds)
            if seed_idx % accelerator.num_processes == accelerator.process_index
        ]

        batch_size = max(1, min(args.seed_concurrency, len(local_seeds)))
        if args.verbose: print('Local seeds:', local_seeds)

        local_rewards = []

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_seeds = local_seeds[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_seeds):
                    thread_args.append((seed_idx, seed, model_list[thread_id], tokenizer, accelerator, thread_id, args.verbose, dataset))

                results = list(executor.map(process_seed, thread_args))
                local_rewards.extend(results)

            force_memory_cleanup()

        all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)
        for seed_idx, reward in local_rewards:
            all_rewards[seed_idx] = reward

        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)
            # # new
            # print(f"[rank {accelerator.process_index}] about to all_reduce on device {torch.cuda.current_device()}")

        rewards = all_rewards.cpu().tolist()
        del all_rewards
        force_memory_cleanup()

        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        original_model = model_list[0]
        for _, param in original_model.named_parameters():
            update = torch.zeros_like(param)
            gen = torch.Generator(device=param.device)
            for seed_idx in range(POPULATION_SIZE):
                gen.manual_seed(int(seeds[seed_idx]))
                noise = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype,
                )
                noise.mul_(float(rewards_normalized[seed_idx]))
                update.add_(noise)
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for model_idx in range(1, len(model_list)):
            target_model = model_list[model_idx]
            for name, param in target_model.named_parameters():
                param.data.copy_(original_model.get_parameter(name).data.clone())

        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time

        mean_reward = rewards_tensor.mean().item()
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()

        del rewards_tensor, rewards_normalized
        force_memory_cleanup()

        if accelerator.is_main_process:
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean: {mean_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")

            if args.log_sample_every > 0 and (iteration + 1) % args.log_sample_every == 0 and dataset_size > 0:
                sample_index = min(dataset_size // 2, dataset_size - 1)
                sample_entry = dataset[sample_index]
                preview_rewards, preview_outputs = evaluate_model(
                    original_model,
                    tokenizer,
                    [sample_entry['prompt']],
                    [sample_entry['target']],
                    accelerator,
                    return_text=True,
                )
                preview_reward = preview_rewards[0] if preview_rewards else float('nan')
                preview_output = preview_outputs[0] if preview_outputs else ''

                prompt_excerpt = sample_entry['prompt'][:200]
                output_excerpt = preview_output[:200]
                print(f"[Preview] Iter {iteration + 1} | Task {sample_entry['task_id']} | Reward {preview_reward:.3f}")
                print(f"[Preview Prompt] {prompt_excerpt}{'...' if len(sample_entry['prompt']) > 200 else ''}")
                print(f"[Preview Output] {output_excerpt}{'...' if len(preview_output) > 200 else ''}")

            if (iteration + 1) % 100 == 0:
                save_model_checkpoint(original_model, tokenizer, iteration + 1, model_name, initial_seed, args, dataset_size)

    total_time = time.time() - training_start_time

    if accelerator.is_main_process:
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        save_model_checkpoint(original_model, tokenizer, NUM_ITERATIONS, model_name, initial_seed, args, dataset_size, tag='final')
        print('Final model saved successfully.')

if __name__ == "__main__":
    os.environ['PYTHONWARNINGS'] = 'ignore'
    mp.set_start_method('spawn', force=True)
    main()
