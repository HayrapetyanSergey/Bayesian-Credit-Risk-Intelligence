
import json

from agent import BayesianCreditRiskAgent


def main():
    agent = BayesianCreditRiskAgent()

    questions = [
        "What is the overall default rate in the dataset?",
        "Compare the posterior default probabilities of all three risk groups.",
        "For 500 customers with a delay of two or more months, predict defaults and give the 95% predictive interval.",
        "Did our Metropolis-Hastings MCMC converge?",
        "Explain the difference between a credible interval and a posterior predictive interval."
    ]

    for index, question in enumerate(questions, 1):
        print("\n" + "=" * 70)
        print(f"TEST {index}")
        print("=" * 70)

        print("\nQUESTION:")
        print(question)

        result = agent.chat(question)

        print("\nAGENT RESPONSE:")
        print(result["answer"])

        print("\nTOOLS USED:")
        print(
            json.dumps(
                [
                    item["name"]
                    for item in result["tools_used"]
                ],
                indent=2
            )
        )


if __name__ == "__main__":
    main()
