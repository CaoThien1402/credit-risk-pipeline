"""The decision threshold app.py applies to at_application_model.pkl's predict_proba.

Derived in notebooks/06_decision_threshold.ipynb, not chosen by convention. 0.5 is not a
neutral default here: build_pipeline's scale_pos_weight (session 10) reweights the loss
to correct for the ~21.8% base rate, which shifts predict_proba's scores away from true
probabilities as a direct side effect. Verified directly in that notebook - at a
predicted score of ~0.50, the real default rate in that bin is ~24%, not 50%. A threshold
picked by eye, or by a closed-form formula that treats the score as a real probability,
would be wrong for a documented, checked reason rather than an unchecked assumption.

DECISION_THRESHOLD was instead found by sweeping every cutoff against the session 9-11
held-out test set (never used to fit the shipped model) and picking the one that
minimizes real expected cost, priced from each held-out row's own loan_amnt/
loan_int_rate/loan_term_months:
  - cost of approving a defaulter (false negative)  ~= loan_amnt (recovery_rate=0%,
    conservative - this dataset has no collections data to measure a real rate)
  - cost of rejecting a good applicant (false positive) ~= loan_amnt * loan_int_rate *
    loan_term_years (simple interest, no discounting)

loan_int_rate is used only in that backtest, to price historical outcomes whose true
label is already known - it never reaches the model as an input, so this does not
reopen the leakage question from session 8-9 (see CLAUDE.md).

0.53 was chosen over three alternatives also computed in that notebook - Youden's J,
F1-maximizing, and a recall>=90% policy target - because all three treat a missed
default and a wrongly rejected good applicant as equally costly (or optimize a policy
constraint instead of cost), and each produced a higher real-dollar cost per applicant
on the same held-out set. 0.53 is also stable under the one assumption behind it: at a
20-40% recovery rate instead of 0%, the cost-minimizing threshold only moves to 0.565.
"""

DECISION_THRESHOLD = 0.53
