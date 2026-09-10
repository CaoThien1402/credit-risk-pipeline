"""Session 9: build the ColumnTransformer + Pipeline yourself.

Per docs/Credit_Risk_Pipeline_Plan_v3.md (buổi 9): write build_preprocessor() and
build_pipeline() by hand - don't have Claude Code generate this part. The goal is
being able to explain every choice in an interview, not just having working code.

Before writing, answer these two questions in your own words (in the notebook markdown
cell above where this module gets imported, or as a comment here):

1. Why must StandardScaler live *inside* the Pipeline and fit only on X_train (after
   train_test_split), not on the full dataset before splitting?

2. Why OneHotEncoder instead of LabelEncoder for the columns in
   model.features.NOMINAL_CATEGORICAL_COLS? What would LabelEncoder wrongly imply to a
   Logistic Regression model that OneHotEncoder doesn't?
"""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Return a ColumnTransformer: StandardScaler on numeric_cols, OneHotEncoder on
    categorical_cols."""
    raise NotImplementedError("Write this yourself - see the module docstring.")


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Wrap build_preprocessor(...) and a LogisticRegression(class_weight='balanced')
    into one Pipeline, so preprocessing is always fit exactly once, on train data only."""
    raise NotImplementedError("Write this yourself - see the module docstring.")
