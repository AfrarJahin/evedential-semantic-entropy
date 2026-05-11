import pickle

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

run_dir = r'EXP/wandb/offline-run-20260507_021501-30yoxokf/files'

val = load(f'{run_dir}/validation_generations.pkl')
train = load(f'{run_dir}/train_generations.pkl')
unc = load(f'{run_dir}/uncertainty_measures.pkl')
details = load(f'{run_dir}/experiment_details.pkl')

args = details.get('args', {})
print('=== EXPERIMENT CONFIG ===')
print(f'Model:           {getattr(args, "model_name", "?")}')
print(f'Dataset:         {getattr(args, "dataset", "?")}')
print(f'Num samples:     {getattr(args, "num_samples", "?")}')
print(f'Num generations: {getattr(args, "num_generations", "?")}')
print(f'Max new tokens:  {getattr(args, "model_max_new_tokens", "?")}')

print()
print('=== VALIDATION RESULTS ===')
print(f'Total questions answered: {len(val)}')
correct = 0
for i, (qid, data) in enumerate(val.items()):
    print(f'\n--- Q{i+1} ---')
    print(f'Question:       {data["question"]}')
    print(f'Correct answer: {data.get("correct_answer", "?")}')
    responses = data.get('responses', [])
    for j, r in enumerate(responses):
        ans = r.get('response', r) if isinstance(r, dict) else r
        print(f'  Gen {j+1}: {ans}')
    acc = data.get('accuracy', None)
    if acc is not None:
        print(f'  Accuracy: {acc}')
        correct += acc

print(f'\nOverall accuracy: {correct}/{len(val)} = {correct/len(val)*100:.1f}%')

print()
print('=== UNCERTAINTY MEASURES ===')
for k, v in unc.items():
    if hasattr(v, '__len__'):
        vals = [round(float(x), 4) if hasattr(x, '__float__') else x for x in v]
        print(f'  {k}: {vals}')
    else:
        try:
            print(f'  {k}: {round(float(v), 4)}')
        except:
            print(f'  {k}: {v}')
