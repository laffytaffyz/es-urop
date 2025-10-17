import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import copy
import os
import argparse
from accelerate import Accelerator
import time
import torch.multiprocessing as mp
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import math
import gc
from accelerate.utils import broadcast_object_list

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='/path/to/hf_cache')
parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'], help='Mixed precision mode')
parser.add_argument('--gpu_threads', type=int, default=4, help='Number of parallel threads per GPU')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
parser.add_argument('--sync_every', type=int, default=20, help='Broadcast model weights every N iterations')
parser.add_argument('--aggressive_gc', action='store_true', help='Perform aggressive garbage collection')
parser.add_argument('--eval_interval', type=int, default=20, help='Interval for evaluating best/worst models')
parser.add_argument('--skip_eval', action='store_true', help='Skip best/worst model evaluation to save memory')
parser.add_argument('--visualization_dir', type=str, default='./visualizations', help='Directory for saving visualizations')
parser.add_argument('--weight_sample_interval', type=int, default=10, help='Sample interval for weight tracking')
args = parser.parse_args()


NUM_ITERATIONS = 1000              # Number of ES iterations (generations)
POPULATION_SIZE = 30              # Population size (number of mutants per iteration)
SIGMA = 0.001                     # Standard deviation for weight perturbations (choose carefully)
ALPHA = 0.0005                    # Learning rate
max_new_tokens = 100
do_sample = False


# --- Dummy Dataset and Reward Function ---
# In practice, define a set of input reasoning tasks with desired targets.
dataset = [
    ("Solve: 3 + 5 =", "8"),
    #("Solve: 21 + 15 =", "36"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
    #("Who is the greatest basketball player?", "Michael Jordan")
    # Add additional tasks as needed.
]

def compute_reward(generated_text, target_text):
    """
    A dummy reward function.
    Replace this with a metric that evaluates the correctness or quality of reasoning.
    """
    # Example: negative absolute difference in length (for demonstration only)
    return -abs(len(generated_text) - len(target_text)) * 0.5

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def evaluate_model(model, tokenizer, input_text, target_text, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False):
    """
    Generate a response from the model given an input (single or batch) and compute rewards.
    """
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")
    
    # Handle both single input and batch input
    is_batch = isinstance(input_text, list)
    input_texts = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]
    
    # Batch tokenization
    input_ids = tokenizer(input_texts, return_tensors="pt", padding=True).input_ids.to(accelerator.device)
    with torch.inference_mode():
        unwrapped_model = accelerator.unwrap_model(model)
        outputs = unwrapped_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)
    
    # Decode batch outputs
    generated_texts = []
    for i in range(len(input_texts)):
        try:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        generated_texts.append(generated_text)
    
    del input_ids, outputs
    torch.cuda.empty_cache()
    
    if verbose and accelerator.is_main_process and thread_id == 0:
        print(generated_texts[0] if generated_texts else "")
    
    # Compute rewards for batch texts
    rewards = [compute_reward(gen_text, tgt_text) for gen_text, tgt_text in zip(generated_texts, target_texts)]
    
    if verbose and accelerator.is_main_process and thread_id == 0:
        print(rewards[0] if rewards else 0)
    
    # Return format based on input type
    if not is_batch:
        if return_text:
            return rewards[0], generated_texts[0]
        return rewards[0]
    else:
        if return_text:
            return rewards, generated_texts
        return rewards

def process_seed(seed_args):
    """Function to process a single seed, used for thread pool"""
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose = seed_args
    
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")
    
    # Put load-evaluate-restore in the same lock block for thread safety
    total_reward = 0.0
    original_model = accelerator.unwrap_model(model)
    for name, param in original_model.named_parameters():
        gen = torch.Generator(device=param.device)

        gen.manual_seed(int(seed))

        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(SIGMA * noise)
    
    # Ensure weights are fully loaded before evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)
    
    # Evaluate all prompts with perturbed weights in batch
    input_texts = [input_text for input_text, _ in dataset]
    target_texts = [target_text for _, target_text in dataset]
    rewards = evaluate_model(model, tokenizer, input_texts, target_texts, accelerator, 
                           seed_idx=seed_idx, thread_id=thread_id, verbose=verbose)
    total_reward = sum(rewards)
    
    # Restore original weights (direct inplace modification)
    for name, param in original_model.named_parameters():
        gen = torch.Generator(device=param.device)

        gen.manual_seed(int(seed))

        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(-SIGMA * noise)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)
    
    average_reward = total_reward / len(dataset)
    
    force_memory_cleanup()
    
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {average_reward:.4f}")
    
    return seed_idx, average_reward


# --- Main Evolution Strategies Loop ---
def main():
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    
    if accelerator.is_main_process:
        print(f"Total processes: {accelerator.num_processes}, GPU threads per process: {args.gpu_threads}")
        print(f"Using mixed precision: {args.mixed_precision}")
        print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
        print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")
        print(f"Model weight sync frequency: every {args.sync_every} iterations")
        print(f"Evaluation interval: {args.eval_interval if not args.skip_eval else 'Disabled'}")
        print(f"Aggressive GC: {args.aggressive_gc}")
        print(f"Visualization directory: {args.visualization_dir}")
    
    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir
    
    if accelerator.is_main_process:
        print(f"Loading model {model_name}...")
    
    model_list = []
    for model_index in range(args.gpu_threads):
        model_list.append(AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            device_map={"": accelerator.process_index},  # Assign devices explicitly
            torch_dtype=torch.float16 if args.mixed_precision == 'fp16' else (torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float32),
            attn_implementation="flash_attention_2"
        ))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)
    
    if accelerator.is_main_process:
        print("Model loaded successfully")
    
    # for model in model_list:
    #     model.eval()  # Turn off dropout, etc.
    #     model = accelerator.prepare(model)
    for i, model in enumerate(model_list):
        model.eval()
        model_list[i] = accelerator.prepare(model)
    
    force_memory_cleanup()
    
    training_start_time = time.time()
    
    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        
        force_memory_cleanup()
        
        if args.verbose:
            print(f"Process {accelerator.process_index} starting iteration {iteration + 1}/{NUM_ITERATIONS}")
        
        # # Generate seeds on main process only
        # if accelerator.is_main_process:
        #     if args.verbose:
        #         print(f"Main process {accelerator.process_index} generating seeds")
        #     seeds = np.random.randint(0, 2**31, size=POPULATION_SIZE, dtype=np.int64).tolist()
        #     seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        # else:
        #     if args.verbose:
        #         print(f"Worker process {accelerator.process_index} waiting for seeds")
        #     seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)
            
        # # Broadcast seeds from main process to all processes
        # torch.distributed.broadcast(seeds_tensor, src=0)
        # seeds = seeds_tensor.cpu().tolist()  # Convert back to list for all processes

        if accelerator.is_main_process:
            if args.verbose:
                print(f"Main process {accelerator.process_index} generating seeds")
            seeds = np.random.randint(0, 2**31, size=POPULATION_SIZE, dtype=np.int64).tolist()
        else:
            if args.verbose:
                print(f"Worker process {accelerator.process_index} waiting for seeds")
            seeds = None

        # Broadcast as a Python object
        obj_list = [seeds]
        broadcast_object_list(obj_list, src=0)
        seeds = obj_list[0]                      # now all ranks have the same list
        seeds_tensor = torch.tensor(             # optional tensor form if you still want it
            seeds, device=accelerator.device, dtype=torch.long
        )
        
        if args.verbose:
            print(f"Process {accelerator.process_index} received seeds")
        
        # Assign seeds to each process for processing
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            # Simple task assignment: assign seeds by process ID
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))
                
        if args.verbose:
            print(f"Process {accelerator.process_index} assigned {len(local_seeds)} seeds: {[idx for idx, _ in local_seeds]}")
        
        # Process seeds in smaller batches to reduce memory pressure
        local_rewards = []
        batch_size = max(1, min(args.gpu_threads, len(local_seeds)))
        
        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_seeds = local_seeds[batch_start:batch_end]
            
            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                # Prepare thread arguments
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_seeds):
                    # Pass verbose flag as argument to process_seed function
                    thread_args.append((seed_idx, seed, model_list[thread_id], tokenizer, accelerator, thread_id, args.verbose))
                
                # Execute in parallel and collect results
                results = list(executor.map(process_seed, thread_args))
                local_rewards.extend(results)
            force_memory_cleanup()
        
        all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)
        
        for seed_idx, reward in local_rewards:
            all_rewards[seed_idx] = reward
            
        # torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)
        all_rewards = accelerator.reduce(all_rewards, reduction="sum")
        
        rewards = all_rewards.cpu().tolist()
        del all_rewards
        force_memory_cleanup()
        
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        

        if args.verbose:
            print(f"Process {accelerator.process_index} updating model weights")
        original_model = accelerator.unwrap_model(model_list[0])
        for name, param in original_model.named_parameters():
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(POPULATION_SIZE):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed))
                
                noise = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype
                )
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            torch.cuda.empty_cache()

        for model_idx in range(1, len(model_list)):
            original_model_tmp = accelerator.unwrap_model(model_list[model_idx])
            for name, param in original_model_tmp.named_parameters():
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
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")
        
        # Save checkpoint every 100 iterations
        if (iteration + 1) % 100 == 0 and accelerator.is_main_process:
            checkpoint_dir = f"checkpoint_iter{iteration + 1}_pop{POPULATION_SIZE}_sigma{SIGMA}_alpha{ALPHA}_{args.mixed_precision}_threads{args.gpu_threads}_max_new_tokens{max_new_tokens}_newnew"
            print(f"Saving checkpoint at iteration {iteration + 1} to {checkpoint_dir}...")
            original_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved successfully.")
    
    total_time = time.time() - training_start_time
    

    if accelerator.is_main_process:
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        save_dir = f"finetuned_qwen_es_random_seed_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_{args.mixed_precision}_threads{args.gpu_threads}"
        print(f"Saving model to {save_dir}...")
        original_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model saved successfully.")
        print(f"Visualizations saved to {args.visualization_dir}")

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()