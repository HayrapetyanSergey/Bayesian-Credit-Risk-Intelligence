
import json

from knowledge_base import (
    get_dataset_summary,
    get_risk_group_posterior,
    predict_defaults,
    compare_risk_groups,
    get_model_diagnostics
)


def show(title, result):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )


def main():
    show(
        "DATASET SUMMARY",
        get_dataset_summary()
    )

    show(
        "BAYESIAN POSTERIOR",
        get_risk_group_posterior()
    )

    show(
        "PREDICTION: 500 CUSTOMERS",
        predict_defaults(
            risk_group="2+ months delay",
            future_customers=500
        )
    )

    show(
        "RISK GROUP COMPARISON",
        compare_risk_groups()
    )

    show(
        "MCMC DIAGNOSTICS",
        get_model_diagnostics()
    )


if __name__ == "__main__":
    main()
