# evaluate_rag.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# Define few QA samples (manually created)
data = {
    "question": ["Quotes about success by Einstein"],
    "contexts": [["success is not final, failure is not fatal"]],
    "answers": ["Success is not final, failure is not fatal."],
    "ground_truths": [["Success is not final, failure is not fatal."]],
}
eval_dataset = Dataset.from_dict(data)

results = evaluate(
    eval_dataset,
    metrics=[faithfulness, answer_relevancy]
)

print(results)
