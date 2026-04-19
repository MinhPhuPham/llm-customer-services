# ===========================================================
# excel_parser.py — Read Excel / Google Sheets → flatten bilingual rows
# ===========================================================
"""
Reads the bilingual training data from Excel or Google Sheets and produces:
  - A list of (text, tag) training pairs with language prefixes
  - A DataFrame with answer templates per tag

Supported data sources:
  1. Local/Drive Excel file (.xlsx)
  2. Google Sheets URL (requires gspread, auto-available in Colab)
  3. Google Sheets by name (requires gspread + auth)
"""

import pandas as pd

from scripts.config import LANG_PREFIX


# ===========================================================
# CORE: Flatten DataFrame → training rows
# ===========================================================
def _flatten_to_train_rows(df):
    """
    Convert a 6-column DataFrame to training pairs with language prefix.

    Expects columns: ['tag', 'type', 'q_en', 'q_ja', 'a_en', 'a_ja']
    """
    # Forward-fill tag and type for grouped rows
    df['tag'] = df['tag'].ffill()
    df['type'] = df['type'].ffill()

    # Forward-fill answers within each tag group
    df['a_en'] = df.groupby('tag')['a_en'].ffill()
    df['a_ja'] = df.groupby('tag')['a_ja'].ffill()

    # Flatten to training rows with language prefix
    train_rows = []
    for _, row in df.iterrows():
        tag = row['tag']

        # English question
        if pd.notna(row['q_en']) and str(row['q_en']).strip():
            text = f"{LANG_PREFIX['en']} {str(row['q_en']).strip()}"
            train_rows.append((text, tag))

        # Japanese question
        if pd.notna(row['q_ja']) and str(row['q_ja']).strip():
            text = f"{LANG_PREFIX['ja']} {str(row['q_ja']).strip()}"
            train_rows.append((text, tag))

    # Summary
    en_count = sum(1 for t, _ in train_rows if t.startswith(LANG_PREFIX['en']))
    ja_count = sum(1 for t, _ in train_rows if t.startswith(LANG_PREFIX['ja']))
    print(f"  Parsed: {len(train_rows)} samples ({en_count} EN + {ja_count} JA)")

    return train_rows, df


# ===========================================================
# SOURCE 1: Excel file (.xlsx) on local/Drive
# ===========================================================
def parse_excel(excel_path):
    """
    Parse the bilingual Excel training data template.

    Expected Excel layout (multi-level header):
        tag | type | Question (English, Japanese) | Answer (English, Japanese)

    Args:
        excel_path: Path to the .xlsx file.

    Returns:
        train_rows: List of (prefixed_text, tag) tuples.
        df: Full DataFrame with forward-filled answers.
    """
    print(f"  Source: Excel file → {excel_path}")

    # Read with multi-level header (row 0 + row 1)
    df = pd.read_excel(excel_path, header=[0, 1])
    df.columns = ['tag', 'type', 'q_en', 'q_ja', 'a_en', 'a_ja']

    return _flatten_to_train_rows(df)


# ===========================================================
# SOURCE 2: Google Sheets (via URL or sheet name)
# ===========================================================
def parse_google_sheet(sheet_url=None, sheet_name=None):
    """
    Parse training data from a Google Sheet.

    The sheet must have the same layout as the Excel template:
        Row 1: tag  | type | Question |          | Answer  |
        Row 2:      |      | English  | Japanese | English | Japanese
        Row 3+: data rows

    Requires: pip install gspread google-auth
    In Colab, both are pre-installed. Call google.colab.auth.authenticate_user()
    before using this function.

    Args:
        sheet_url: Full Google Sheets URL (e.g., https://docs.google.com/spreadsheets/d/xxx/edit)
        sheet_name: Sheet name in Google Drive (alternative to URL).
                    Exactly one of sheet_url or sheet_name must be provided.

    Returns:
        train_rows: List of (prefixed_text, tag) tuples.
        df: Full DataFrame with forward-filled answers.
    """
    try:
        import gspread
        from google.auth import default as google_auth_default
    except ImportError:
        raise ImportError(
            "Google Sheets support requires gspread and google-auth.\n"
            "Install: pip install gspread google-auth\n"
            "In Colab, run: from google.colab import auth; auth.authenticate_user()"
        )

    # Authenticate
    creds, _ = google_auth_default(
        scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )
    gc = gspread.authorize(creds)

    # Open sheet
    if sheet_url:
        print(f"  Source: Google Sheet URL → {sheet_url}")
        spreadsheet = gc.open_by_url(sheet_url)
    elif sheet_name:
        print(f"  Source: Google Sheet name → {sheet_name}")
        spreadsheet = gc.open(sheet_name)
    else:
        raise ValueError("Provide either sheet_url or sheet_name")

    # Read first worksheet
    worksheet = spreadsheet.sheet1
    all_values = worksheet.get_all_values()

    if len(all_values) < 3:
        raise ValueError(
            f"Sheet has only {len(all_values)} rows. "
            f"Expected at least 3 (2 header rows + 1 data row)."
        )

    # Skip 2 header rows, assign our column names
    data_rows = all_values[2:]  # skip row 0 (level-1 header) + row 1 (level-2 header)
    df = pd.DataFrame(data_rows, columns=['tag', 'type', 'q_en', 'q_ja', 'a_en', 'a_ja'])

    # Replace empty strings with NaN for consistent handling
    df = df.replace('', pd.NA)

    return _flatten_to_train_rows(df)
