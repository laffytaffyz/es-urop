import os, json

data_path = os.path.join(os.path.dirname(__file__), '/orcd/data/tpoggio/001/tiffany8/es-urop/data/arc-prize-2025/arc-agi_training_challenges.json')
solution_path = os.path.join(os.path.dirname(__file__), '/orcd/data/tpoggio/001/tiffany8/es-urop/data/arc-prize-2025/arc-agi_training_solutions.json')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset file not found: {data_path}")

with open(data_path, 'r') as f:
    data_json = json.load(f)

with open(solution_path, 'r') as f:
    sol_json = json.load(f)

dataset = []
for item in data_json:
    examples = data_json[item]['train']
    query = data_json[item]['test']
    target = sol_json[item]
    example_str = "\n".join([f"Input: {example['input']}\nOutput: {example['output']}" for example in examples])
    prompt = f"""
            Given the input / output examples, provide step by step instructions for how to transform the input grids into output grids.
            Examples:
            {example_str}
            
            Question:
            Input:
            {query}
            """

    dataset.append({'examples' : examples, 
                    'query' : query, 
                    'target' : target,
                    'prompt': prompt})

print(dataset[0])


example_response = """



"""