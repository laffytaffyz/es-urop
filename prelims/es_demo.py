import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import copy
import os
import argparse

logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_device', type=str, default='0,1')
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='/opt/dlami/nvme/vsonicv/huggingface_cache')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_device

# Evolution Strategies hyperparameters
NUM_ITERATIONS = 1000              # Number of ES iterations (generations)
POPULATION_SIZE = 30              # Population size (number of mutants per iteration)
SIGMA = 0.001                      # Standard deviation for weight perturbations (choose carefully)
ALPHA = 0.0005                    # Learning rate
max_new_tokens = 200
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
    return -abs(len(generated_text) - len(target_text))

def evaluate_model(model, tokenizer, input_text, target_text):
    """
    Generate a response from the model given an input and compute a reward.
    """
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample)
    #generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except TypeError:
        # fallback: manually convert ids→tokens, filter None, then join
        tokens = tokenizer.convert_ids_to_tokens(outputs[0], skip_special_tokens=True)
        filtered = [t for t in tokens if t is not None]
        generated_text = tokenizer.convert_tokens_to_string(filtered)
    reward = compute_reward(generated_text, target_text)
    return reward

# --- Main Evolution Strategies Loop ---
def main():
    # Load large language model
    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir
    model = AutoModelForCausalLM.from_pretrained(model_name,
    #model = AutoModelForCausalLM.from_pretrained("finetuned_qwen_es_random_seed_pop30_iter1000_sigma0.001_alpha0.0005",
                                                torch_dtype=torch.float16,
                                                cache_dir=hf_cache_dir, device_map="auto")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)
    
    model.eval()  # Turn off dropout, etc.
    
    # Get the initial full set of parameters as a state dictionary
    base_state = copy.deepcopy(model.state_dict())
    #print(base_state.items())

    
    for iteration in range(NUM_ITERATIONS):
        rewards = []
        noise_samples = []  # List of dictionaries; each dict maps parameter names to noise tensors
        # Sample one seed per population member
        seeds = np.random.randint(0, 2**31, size=POPULATION_SIZE, dtype=np.int64).tolist()
        
        for i, seed in enumerate(seeds):
            perturbed_state = {}
            #gen = torch.Generator()  # CPU‐side generator
            for name, base_param in base_state.items():
                gen = torch.Generator(device=base_param.device)
                # re-seed and generate exactly the same noise every time
                gen.manual_seed(int(seed))
                # draw noise in the same device/shape as the parameter
                noise = torch.randn(
                    base_param.shape,
                    generator=gen,
                    device=base_param.device,
                    dtype=base_param.dtype
                )
                perturbed_state[name] = base_param + SIGMA * noise

            model.load_state_dict(perturbed_state)
            # Evaluate on the dataset and compute average reward.
            total_reward = 0.0
            for (input_text, target_text) in dataset:
                reward = evaluate_model(model, tokenizer, input_text, target_text)
                total_reward += reward
            average_reward = total_reward / len(dataset)
            rewards.append(average_reward)

        # After evaluation, restore the base weights.
        model.load_state_dict(base_state)
        
        # Convert rewards to a tensor and normalize.
        #rewards_tensor = torch.tensor(rewards, device="cuda")
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        #print("rewards_normalized: {}".format(rewards_normalized))
        
        # now build aggregated_update with only seeds + rewards_norm
        aggregated_update = {}
        for name, base_param in base_state.items():
            update = torch.zeros_like(base_param)
            gen = torch.Generator(device=base_param.device)
            for r_norm, seed in zip(rewards_normalized, seeds):
                gen.manual_seed(int(seed))
                noise = torch.randn(
                    base_param.shape,
                    generator=gen,
                    device=base_param.device,
                    dtype=base_param.dtype
                )
                update += noise * float(r_norm)
            update /= POPULATION_SIZE
            aggregated_update[name] = update
        
        # Update base weights using the ES update rule.
        for name in base_state.keys():
            #base_state[name] = base_state[name] + (ALPHA / SIGMA) * aggregated_update[name]
            base_state[name] = base_state[name] + ALPHA * aggregated_update[name]
        
        # Load the updated weights back into the model.
        model.load_state_dict(base_state)
        
        # Log the progress
        mean_reward = rewards_tensor.mean().item()
        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Mean Reward: {mean_reward:.4f}")
    
    # After ES, save the fine-tuned model weights.
    model.save_pretrained("finetuned_qwen_es_random_seed_pop{}_iter{}_sigma{}_alpha{}".format(POPULATION_SIZE, NUM_ITERATIONS, SIGMA, ALPHA))
    tokenizer.save_pretrained("finetuned_qwen_es")
    print("Evolution strategies fine-tuning complete. Fine-tuned model saved to 'finetuned_qwen_es'.")

if __name__ == "__main__":
    main()