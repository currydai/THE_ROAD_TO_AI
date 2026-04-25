from agent_from_zero.evaluation.datasets import load_demo_dataset
from agent_from_zero.evaluation.evaluators import has_citation, keyword_relevance
from agent_from_zero.rag.qa_chain import answer_question


def main() -> None:
    results = []
    for example in load_demo_dataset():
        answer = answer_question(example.question).answer
        results.append((example.question, keyword_relevance(answer, example.expected_keywords)))
        results.append((example.question, has_citation(answer)))

    for question, result in results:
        print(f"[{result.name}] {result.score:.2f} - {question} - {result.comment}")


if __name__ == "__main__":
    main()
