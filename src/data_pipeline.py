from pathlib import Path
import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

SYMPTOMS_FILE = RAW_DATA_DIR / "aiml_healthcare_symptoms.csv"
TEST_CASES_FILE = RAW_DATA_DIR / "aiml_healthcare_test_cases.txt"


def load_symptoms():
    """Load the healthcare symptoms dataset."""
    df = pd.read_csv(SYMPTOMS_FILE)

    print(f"Loaded symptoms dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_test_cases():
    """Load the clinical test notes."""
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as file:
        text = file.read()

    print(f"Loaded test cases: {len(text)} characters")
    return text


def save_processed_data(df):
    """Save the processed symptoms dataset."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_DATA_DIR / "symptoms_processed.csv"
    df.to_csv(output_file, index=False)

    print(f"Processed data saved to: {output_file}")


def main():
    print("=== CareBot Data Pipeline ===")

    symptoms_df = load_symptoms()
    test_cases = load_test_cases()

    print("\nSymptoms dataset preview:")
    print(symptoms_df.head())

    print("\nTest cases preview:")
    print(test_cases[:500])

    save_processed_data(symptoms_df)

    print("\nData pipeline completed successfully!")


if __name__ == "__main__":
    main()