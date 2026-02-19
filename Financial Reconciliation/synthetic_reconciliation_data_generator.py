import os
import json
import random
import string
import math
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from faker import Faker
except ImportError:
    raise SystemExit("Please install Faker first: pip install Faker")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    raise SystemExit("Please install pandas and numpy: pip install pandas numpy")


# ==========================
# CONFIGURATION
# ==========================

CONFIG = {
    "root_output_dir": "data",
    "seed": 42,
    "n_vendors": 10,
    "n_customers": 100,
    "n_invoices": 200,          # scale here
    "n_receipts": 150,
    "max_line_items_per_doc": 8,
    "bank_txn_base_multiplier": 1.4,  # bank txns ~ 1.4x docs to allow many/no-match
    "date_range_days": 60,       # lookback window --> Make it to 365 for more data
    "currency_list": ["USD", "EUR", "GBP", "AED"],
    "missing_invoice_rate": 0.08,  # invoice not in bank
    "missing_bank_rate": 0.05,     # bank txn without invoice
    "partial_match_rate": 0.10,    # amount mismatches, partial payments, etc.
    "multi_to_one_rate": 0.06,     # multiple invoices -> one bank payment
    "one_to_multi_rate": 0.04,     # one invoice -> multiple bank entries (split)
    "ocr_noise_rate": 0.15,        # probability field gets OCR noise
    "ocr_dropout_rate": 0.05,      # probability field is dropped
    "ocr_typo_rate": 0.20,         # probability of typos in strings
}


# ==========================
# UTILS
# ==========================

fake = Faker()
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


def ensure_dirs(root):
    subdirs = [
        "output/invoices/ocr_noise",
        "output/bank",
        "output/reconciliation",
        "output/metadata",
    ]
    for sd in subdirs:
        os.makedirs(os.path.join(root, sd), exist_ok=True)


def random_currency():
    return random.choice(CONFIG["currency_list"])


def random_date_within_days(days_back):
    base = datetime.now()
    delta_days = random.randint(0, days_back)
    return base - timedelta(days=delta_days)


def add_noise_to_string(s, typo_prob=0.2):
    """Inject random OCR typos: character swaps, drops, inserts."""
    if not s:
        return s
    s = str(s)
    if random.random() > typo_prob:
        return s

    operations = ["swap", "drop", "insert"]
    op = random.choice(operations)

    if op == "swap" and len(s) > 1:
        idx = random.randint(0, len(s) - 2)
        lst = list(s)
        lst[idx], lst[idx + 1] = lst[idx + 1], lst[idx]
        return "".join(lst)
    elif op == "drop" and len(s) > 1:
        idx = random.randint(0, len(s) - 1)
        return s[:idx] + s[idx + 1 :]
    else:  # insert
        idx = random.randint(0, len(s))
        ch = random.choice(string.ascii_letters + string.digits)
        return s[:idx] + ch + s[idx:]


def maybe_dropout(value, dropout_rate):
    if random.random() < dropout_rate:
        return None
    return value


def amount_with_small_noise(amount, max_pct=0.1):
    """Introduce small percentage difference to simulate partial payments, fees, FX, etc."""
    delta = amount * random.uniform(-max_pct, max_pct)
    return round(amount + delta, 2)


# ==========================
# MASTER DATA
# ==========================

def generate_vendors(n):
    vendors = []
    for vid in range(1, n + 1):
        vendors.append(
            {
                "vendor_id": f"V{vid:05d}",
                "vendor_name": fake.company(),
                "country": fake.country(),
                "city": fake.city(),
                "iban": fake.iban(),
            }
        )
    return vendors


def generate_customers(n):
    customers = []
    for cid in range(1, n + 1):
        customers.append(
            {
                "customer_id": f"C{cid:05d}",
                "customer_name": fake.name(),
                "segment": random.choice(["SMB", "Enterprise", "Individual"]),
                "country": fake.country(),
                "city": fake.city(),
            }
        )
    return customers


def generate_line_items(doc_id, max_items):
    n_items = random.randint(1, max_items)
    items = []
    for i in range(1, n_items + 1):
        qty = max(1, int(np.random.exponential(2)))
        unit_price = round(np.random.lognormal(mean=2.5, sigma=0.7), 2)
        discount_pct = random.choice([0, 0, 0, 5, 10, 15])
        line_amount = round(qty * unit_price * (1 - discount_pct / 100.0), 2)

        items.append(
            {
                "doc_id": doc_id,
                "line_no": i,
                "description": fake.catch_phrase(),
                "quantity": qty,
                "unit_price": unit_price,
                "discount_pct": discount_pct,
                "line_amount": line_amount,
            }
        )
    return items


def compute_header_totals(line_items):
    subtotal = sum(li["line_amount"] for li in line_items)
    tax_rate = random.choice([0, 5, 5, 10, 15])
    tax_amount = round(subtotal * tax_rate / 100.0, 2)
    shipping = round(random.choice([0, 0, 5, 10, 20]), 2)
    total = round(subtotal + tax_amount + shipping, 2)
    return subtotal, tax_rate, tax_amount, shipping, total


# ==========================
# DOCUMENT (INVOICE/RECEIPT) GENERATION
# ==========================

def generate_docs(doc_type, n_docs, vendors, customers):
    """
    doc_type: 'INV' or 'RCT'
    """
    headers = []
    all_line_items = []

    for i in range(1, n_docs + 1):
        doc_id = f"{doc_type}-{i:07d}"
        vendor = random.choice(vendors)
        customer = random.choice(customers)
        issue_date = random_date_within_days(CONFIG["date_range_days"])
        due_date = issue_date + timedelta(days=random.choice([7, 14, 30, 45, 60]))
        currency = random_currency()

        line_items = generate_line_items(doc_id, CONFIG["max_line_items_per_doc"])
        subtotal, tax_rate, tax_amount, shipping, total = compute_header_totals(line_items)

        header = {
            "doc_id": doc_id,
            "doc_type": "invoice" if doc_type == "INV" else "receipt",
            "vendor_id": vendor["vendor_id"],
            "vendor_name": vendor["vendor_name"],
            "customer_id": customer["customer_id"],
            "customer_name": customer["customer_name"],
            "issue_date": issue_date.strftime("%Y-%m-%d"),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "currency": currency,
            "subtotal": subtotal,
            "tax_rate": tax_rate,
            "tax_amount": tax_amount,
            "shipping": shipping,
            "total_amount": total,
            "payment_terms": random.choice(
                ["NET7", "NET14", "NET30", "NET45", "DUE_ON_RECEIPT"]
            ),
            "po_number": f"PO-{random.randint(100000, 999999)}",
            "status": random.choice(["OPEN", "PAID", "PARTIALLY_PAID", "VOID"]),
        }

        headers.append(header)
        all_line_items.extend(line_items)

    return headers, all_line_items


# ==========================
# BANK TRANSACTIONS
# ==========================

def generate_bank_transactions_from_docs(doc_headers):
    """
    Start with doc totals and create different match patterns
    (exact matches, partial payments, multi-to-one, one-to-multi, missing).
    """
    bank_txns = []
    reconc_links = []

    # Working pools for pattern allocation
    doc_ids = [h["doc_id"] for h in doc_headers]

    # Determine volumes for different patterns
    n_docs = len(doc_headers)
    n_multi_to_one = int(CONFIG["multi_to_one_rate"] * n_docs)
    n_one_to_multi = int(CONFIG["one_to_multi_rate"] * n_docs)

    chosen_for_multi_to_one = set(random.sample(doc_ids, n_multi_to_one))
    remaining_for_one_to_multi = [d for d in doc_ids if d not in chosen_for_multi_to_one]
    chosen_for_one_to_multi = set(random.sample(remaining_for_one_to_multi, n_one_to_multi))

    doc_lookup = {h["doc_id"]: h for h in doc_headers}

    bank_id_counter = 1

    # Helper to create a bank transaction record
    def create_bank_txn(amount, date, currency, doc_ids_for_desc=None):
        nonlocal bank_id_counter
        tx_id = f"BTX-{bank_id_counter:08d}"
        bank_id_counter += 1
        desc_docs = ""
        if doc_ids_for_desc:
            # simulate reference in description
            chosen = random.sample(doc_ids_for_desc, k=min(len(doc_ids_for_desc), 3))
            desc_docs = " ".join(chosen)
        txn = {
            "bank_txn_id": tx_id,
            "booking_date": date.strftime("%Y-%m-%d"),
            "value_date": (date + timedelta(days=random.choice([-1, 0, 1]))).strftime(
                "%Y-%m-%d"
            ),
            "amount": round(amount, 2),
            "currency": currency,
            "counterparty_name": fake.company(),
            "counterparty_account": fake.iban(),
            "description": f"PAYMENT {desc_docs} REF {fake.bothify(text='???####')}",
            "channel": random.choice(
                ["WIRE", "ACH", "CARD", "CASH", "CHECK", "INTERNAL_TRANSFER"]
            ),
        }
        return txn

    # Multi-to-one: several invoices paid by single bank transaction
    multi_to_one_groups = []
    all_docs_for_multi = list(chosen_for_multi_to_one)
    random.shuffle(all_docs_for_multi)
    while all_docs_for_multi:
        group_size = random.randint(2, 5)
        group = all_docs_for_multi[:group_size]
        all_docs_for_multi = all_docs_for_multi[group_size:]
        if len(group) < 2:
            break
        multi_to_one_groups.append(group)

    for group in multi_to_one_groups:
        total_amount = sum(doc_lookup[d]["total_amount"] for d in group)
        doc_example = doc_lookup[group[0]]
        pay_date = datetime.strptime(doc_example["issue_date"], "%Y-%m-%d") + timedelta(
            days=random.randint(0, 45)
        )
        currency = doc_example["currency"]

        # Add some FX/fee noise
        bank_amount = amount_with_small_noise(total_amount, max_pct=0.05)
        bank_txn = create_bank_txn(bank_amount, pay_date, currency, group)
        bank_txns.append(bank_txn)

        for d in group:
            reconc_links.append(
                {
                    "doc_id": d,
                    "bank_txn_id": bank_txn["bank_txn_id"],
                    "link_type": "multi_to_one",
                }
            )

    # One-to-multi: one invoice paid by multiple bank transactions
    for doc_id in chosen_for_one_to_multi:
        header = doc_lookup[doc_id]
        total = header["total_amount"]
        n_parts = random.randint(2, 4)
        remaining = total
        pay_date = datetime.strptime(header["issue_date"], "%Y-%m-%d")

        parts = []
        for i in range(1, n_parts + 1):
            if i == n_parts:
                part_amount = remaining
            else:
                # ensure amounts aren't trivial
                part_amount = max(1.0, remaining * random.uniform(0.1, 0.7))
                remaining -= part_amount
            parts.append(round(part_amount, 2))

        for part in parts:
            txn_date = pay_date + timedelta(days=random.randint(0, 60))
            bank_txn = create_bank_txn(
                amount_with_small_noise(part, max_pct=0.03),
                txn_date,
                header["currency"],
                [doc_id],
            )
            bank_txns.append(bank_txn)
            reconc_links.append(
                {
                    "doc_id": doc_id,
                    "bank_txn_id": bank_txn["bank_txn_id"],
                    "link_type": "one_to_multi",
                }
            )

    # Remaining docs: some exact matches, some partial, some missing
    already_in_links = {r["doc_id"] for r in reconc_links}
    remaining_docs = [d for d in doc_ids if d not in already_in_links]

    for doc_id in remaining_docs:
        header = doc_lookup[doc_id]
        prob_missing_doc = CONFIG["missing_invoice_rate"]
        prob_partial = CONFIG["partial_match_rate"]

        # Some docs never appear in bank (e.g., still unpaid)
        if random.random() < prob_missing_doc:
            # no bank entry for this doc
            reconc_links.append(
                {"doc_id": doc_id, "bank_txn_id": None, "link_type": "missing_in_bank"}
            )
            continue

        amount = header["total_amount"]
        date = datetime.strptime(header["issue_date"], "%Y-%m-%d") + timedelta(
            days=random.randint(0, 60)
        )

        if random.random() < prob_partial:
            bank_amount = amount_with_small_noise(amount, max_pct=0.15)
            link_type = "partial_or_mismatch"
        else:
            bank_amount = amount
            link_type = "exact"

        bank_txn = create_bank_txn(bank_amount, date, header["currency"], [doc_id])
        bank_txns.append(bank_txn)
        reconc_links.append(
            {"doc_id": doc_id, "bank_txn_id": bank_txn["bank_txn_id"], "link_type": link_type}
        )

    # Extra bank-only transactions (no matching docs) to simulate noise, fees, FX, etc.
    n_extra_bank = int(len(doc_headers) * CONFIG["missing_bank_rate"])
    for _ in range(n_extra_bank):
        amount = round(np.random.lognormal(mean=2.0, sigma=1.0), 2)
        date = random_date_within_days(CONFIG["date_range_days"])
        bank_txn = create_bank_txn(amount, date, random_currency(), None)
        bank_txns.append(bank_txn)

    return bank_txns, reconc_links


# ==========================
# OCR-LIKE NOISY JSON
# ==========================

def generate_ocr_json_for_doc(header, line_items, output_path):
    """
    Writes OCR-like JSON for a document:
    - Fields may be missing or noisy.
    - Simulates text blocks instead of structured fields.
    """
    blocks = []

    def noisy_field(name, value):
        if value is None:
            return None
        val = str(value)
        val = maybe_dropout(val, CONFIG["ocr_dropout_rate"])
        if val is None:
            return None
        val = add_noise_to_string(val, typo_prob=CONFIG["ocr_typo_rate"])
        return val

    header_fields = [
        "doc_id",
        "doc_type",
        "vendor_name",
        "customer_name",
        "issue_date",
        "due_date",
        "currency",
        "total_amount",
        "po_number",
        "payment_terms",
    ]

    # Simulate header blocks
    for field in header_fields:
        raw_val = header.get(field)
        if raw_val is None:
            continue
        text_val = noisy_field(field, raw_val)
        if text_val is None:
            continue
        blocks.append(
            {
                "text": text_val,
                "field_hint": field,
                "bbox": [
                    round(random.random(), 3),
                    round(random.random(), 3),
                    round(random.random(), 3),
                    round(random.random(), 3),
                ],
                "page": 1,
            }
        )

    # Simulate line items text in table-like blocks
    for li in line_items:
        line_text = f"{li['description']} {li['quantity']} x {li['unit_price']} = {li['line_amount']}"
        line_text = noisy_field("line_item", line_text)
        if line_text is None:
            continue
        blocks.append(
            {
                "text": line_text,
                "field_hint": "line_item",
                "bbox": [
                    round(random.random(), 3),
                    round(random.random(), 3),
                    round(random.random(), 3),
                    round(random.random(), 3),
                ],
                "page": random.choice([1, 1, 2]),
            }
        )

    # Random rotation/noise info
    meta = {
        "doc_id": header["doc_id"],
        "scanned_pages": random.randint(1, 3),
        "rotation_degrees": random.choice([0, 0, 0, 90, 180, 270]),
        "dpi": random.choice([200, 300, 300, 300]),
    }

    ocr_obj = {"meta": meta, "blocks": blocks}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_obj, f, indent=2)


# ==========================
# MESSY BANK STATEMENT
# ==========================

def create_messy_bank_statement(bank_df):
    """
    Create a messier variant:
    - shuffled order
    - duplicated entries
    - some missing fields
    - slightly altered descriptions
    """
    df = bank_df.copy()

    # Shuffle
    df = df.sample(frac=1.0, random_state=CONFIG["seed"]).reset_index(drop=True)

    # Random duplicates
    n_dup = int(0.03 * len(df))
    dup_rows = df.sample(n=n_dup, replace=False)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Missing fields
    for col in ["description", "counterparty_name", "channel"]:
        mask = np.random.rand(len(df)) < 0.03
        df.loc[mask, col] = None

    # Slight description mutations
    def mutate_desc(desc):
        if pd.isna(desc):
            return desc
        if random.random() < 0.25:
            return add_noise_to_string(desc, typo_prob=0.3)
        return desc

    df["description"] = df["description"].apply(mutate_desc)

    # Shuffled again
    df = df.sample(frac=1.0, random_state=CONFIG["seed"] + 1).reset_index(drop=True)
    return df


# ==========================
# RECONCILIATION REPORTS
# ==========================

def build_missing_items_report(doc_headers, bank_df, reconc_links):
    """
    Report aimed at accountant:
    - docs missing in bank
    - suspicious partial matches
    - bank-only txns without docs
    """
    doc_lookup = {h["doc_id"]: h for h in doc_headers}
    bank_lookup = {r["bank_txn_id"]: r for _, r in bank_df.iterrows()}

    rows = []

    # from doc perspective
    for link in reconc_links:
        doc_id = link["doc_id"]
        bank_id = link["bank_txn_id"]
        link_type = link["link_type"]
        header = doc_lookup[doc_id]

        if link_type == "missing_in_bank":
            rows.append(
                {
                    "issue": "DOC_WITHOUT_BANK",
                    "doc_id": doc_id,
                    "bank_txn_id": "",
                    "doc_amount": header["total_amount"],
                    "bank_amount": "",
                    "currency": header["currency"],
                    "detail": "Document not found in bank statement (likely unpaid or missing).",
                }
            )
        elif link_type in ["partial_or_mismatch", "one_to_multi", "multi_to_one"]:
            bank_row = bank_lookup.get(bank_id)
            bank_amt = bank_row["amount"] if bank_row is not None else ""
            rows.append(
                {
                    "issue": "POTENTIAL_MISMATCH",
                    "doc_id": doc_id,
                    "bank_txn_id": bank_id or "",
                    "doc_amount": header["total_amount"],
                    "bank_amount": bank_amt,
                    "currency": header["currency"],
                    "detail": f"Mismatched or complex mapping ({link_type}). Requires manual review.",
                }
            )

    # from bank perspective: bank txns without docs
    linked_bank_ids = {l["bank_txn_id"] for l in reconc_links if l["bank_txn_id"] is not None}
    for _, row in bank_df.iterrows():
        if row["bank_txn_id"] not in linked_bank_ids:
            rows.append(
                {
                    "issue": "BANK_WITHOUT_DOC",
                    "doc_id": "",
                    "bank_txn_id": row["bank_txn_id"],
                    "doc_amount": "",
                    "bank_amount": row["amount"],
                    "currency": row["currency"],
                    "detail": "Bank transaction has no matching invoice/receipt.",
                }
            )

    report_df = pd.DataFrame(rows)
    return report_df


def build_many_to_one_cases(reconc_links):
    """
    Extract explicit many-to-one patterns for easier debugging/teaching.
    """
    by_bank = defaultdict(list)
    for link in reconc_links:
        if link["bank_txn_id"]:
            by_bank[link["bank_txn_id"]].append(link["doc_id"])

    rows = []
    for bank_id, docs in by_bank.items():
        if len(docs) > 1:
            rows.append(
                {
                    "bank_txn_id": bank_id,
                    "doc_ids": ",".join(sorted(set(docs))),
                    "n_docs": len(set(docs)),
                }
            )

    return pd.DataFrame(rows)


# ==========================
# METADATA FILES
# ==========================

def write_metadata(root, invoices_header_df, receipts_header_df, bank_df):
    meta_dir = os.path.join(root, "output", "metadata")
    schema_path = os.path.join(meta_dir, "schema_description.md")
    dict_path = os.path.join(meta_dir, "data_dictionary.csv")
    notes_path = os.path.join(meta_dir, "generation_notes.md")

    with open(schema_path, "w", encoding="utf-8") as f:
        f.write("# Schema Description\n\n")
        f.write("## Invoices Header\n")
        f.write(", ".join(invoices_header_df.columns))
        f.write("\n\n## Receipts Header\n")
        f.write(", ".join(receipts_header_df.columns))
        f.write("\n\n## Bank Statement\n")
        f.write(", ".join(bank_df.columns))
        f.write("\n")

    # simple data dictionary
    dict_rows = []
    for col in invoices_header_df.columns:
        dict_rows.append(
            {"table": "invoices_header", "column": col, "description": "See script comments."}
        )
    for col in receipts_header_df.columns:
        dict_rows.append(
            {"table": "receipts_header", "column": col, "description": "See script comments."}
        )
    for col in bank_df.columns:
        dict_rows.append(
            {"table": "bank_statement", "column": col, "description": "See script comments."}
        )
    pd.DataFrame(dict_rows).to_csv(dict_path, index=False)

    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("# Generation Notes\n\n")
        f.write("- Synthetic invoices/receipts generated using Faker, lognormal and exponential distributions.\n")
        f.write("- Bank transactions created with complex mapping patterns: exact, partial, one-to-many, many-to-one, and missing.\n")
        f.write("- OCR JSON adds noise: dropped fields, typos, and random bounding boxes to approximate real scanned documents.\n")
        f.write("- See the script for parameters controlling volumes and noise rates.\n")


# ==========================
# MAIN
# ==========================

def main():
    root = CONFIG["root_output_dir"]
    ensure_dirs(root)

    # Master data
    vendors = generate_vendors(CONFIG["n_vendors"])
    customers = generate_customers(CONFIG["n_customers"])

    # Invoices and receipts
    inv_headers, inv_lines = generate_docs("INV", CONFIG["n_invoices"], vendors, customers)
    rct_headers, rct_lines = generate_docs("RCT", CONFIG["n_receipts"], vendors, customers)

    inv_headers_df = pd.DataFrame(inv_headers)
    inv_lines_df = pd.DataFrame(inv_lines)
    rct_headers_df = pd.DataFrame(rct_headers)
    rct_lines_df = pd.DataFrame(rct_lines)

    # Bank transactions from all docs
    all_doc_headers = inv_headers + rct_headers
    bank_txns, reconc_links = generate_bank_transactions_from_docs(all_doc_headers)
    bank_df = pd.DataFrame(bank_txns)
    reconc_links_df = pd.DataFrame(reconc_links)

    # OCR JSON dumps per doc
    ocr_dir = os.path.join(root, "output", "invoices", "ocr_noise")
    # To keep generation time reasonable, you can subsample here if needed
    per_doc_lines = defaultdict(list)
    for li in inv_lines + rct_lines:
        per_doc_lines[li["doc_id"]].append(li)

    for header in all_doc_headers:
        doc_id = header["doc_id"]
        ocr_path = os.path.join(ocr_dir, f"{doc_id}.json")
        generate_ocr_json_for_doc(header, per_doc_lines[doc_id], ocr_path)

    # Messy bank statement variant
    bank_messy_df = create_messy_bank_statement(bank_df)

    # Reconciliation reports
    missing_report_df = build_missing_items_report(all_doc_headers, bank_df, reconc_links)
    many_to_one_cases_df = build_many_to_one_cases(reconc_links)

    # Write CSVs
    inv_dir = os.path.join(root, "output", "invoices")
    bank_dir = os.path.join(root, "output", "bank")
    recon_dir = os.path.join(root, "output", "reconciliation")

    inv_headers_df.to_csv(os.path.join(inv_dir, "invoices_header.csv"), index=False)
    inv_lines_df.to_csv(os.path.join(inv_dir, "invoices_line_items.csv"), index=False)
    rct_headers_df.to_csv(os.path.join(inv_dir, "receipts_header.csv"), index=False)
    rct_lines_df.to_csv(os.path.join(inv_dir, "receipts_line_items.csv"), index=False)

    bank_df.to_csv(os.path.join(bank_dir, "bank_statement.csv"), index=False)
    bank_messy_df.to_csv(os.path.join(bank_dir, "bank_statement_messy.csv"), index=False)

    reconc_links_df.to_csv(os.path.join(recon_dir, "ground_truth_links.csv"), index=False)
    missing_report_df.to_csv(os.path.join(recon_dir, "missing_items_report.csv"), index=False)
    many_to_one_cases_df.to_csv(
        os.path.join(recon_dir, "many_to_one_mapping_cases.csv"), index=False
    )

    # Metadata
    write_metadata(root, inv_headers_df, rct_headers_df, bank_df)

    print(f"Synthetic dataset generated under: {os.path.abspath(root)}")


if __name__ == "__main__":
    main()
