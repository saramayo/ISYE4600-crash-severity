"""Pooled logistic baseline. Calls run_logistic; saves baseline_results + coef + FN csv."""
from __future__ import annotations

import pandas as pd

import baseline_common as bc
from logistic_regression_baseline import run_logistic


def main() -> None:
    df_known, train_df, test_df = bc.load_and_split_verbose()
    lr_rows, coef_df, fn_df = run_logistic(train_df, test_df, df_known)

    pd.DataFrame(lr_rows).to_csv(bc.BASELINE_DIR / "baseline_results.csv", index=False)
    coef_df.to_csv(bc.LR_DIR / "lr_coefficients.csv", index=False)
    fn_df.to_csv(bc.LR_DIR / "false_negatives.csv", index=False)

    print("\n" + "=" * 70)
    print("Saved (logistic regression baseline)")
    print("=" * 70)
    print(f"   {bc.BASELINE_DIR / 'baseline_results.csv'}")
    print(f"   {bc.LR_DIR / 'lr_coefficients.csv'}")
    print(f"   {bc.LR_DIR / 'false_negatives.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
