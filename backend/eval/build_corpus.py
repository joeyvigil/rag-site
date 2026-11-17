"""Generate the evaluation corpus: documents for a fictional company.

Run: python -m eval.build_corpus

The corpus is deliberately built with traps that naive RAG struggles with,
so later phases have measurable room to improve:

- Superseded versions: an archived 2024 PTO policy contradicts the 2026 handbook.
- Near-duplicates: two project reports with identical structure, different numbers.
- Exact identifiers: SKUs, error codes and ticket IDs, which pure vector search
  handles poorly (hybrid search should help in Phase 3).
- Tables: budget tables in DOCX, rows in CSV (column headers get separated from
  rows by chunking).
- Bulk noise: a large generated product catalog and support-ticket log.
- Semantic distractors: weekly digests about OTHER projects using the same
  vocabulary (budgets, pilots, leads, GA dates) as the real questions.
- Facts spread across documents, for multi-hop questions.

Everything is fictional. The random generator is seeded so the corpus is reproducible.
"""

import csv
import io
import random
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"


# ---------------------------------------------------------------- handbook (PDF)

HANDBOOK_PAGES = [
    """Halcyon Robotics Employee Handbook
Effective January 1, 2026. This handbook replaces all earlier policy documents.

1. Paid Time Off
Full-time employees accrue 22 days of paid time off (PTO) per calendar year. Part-time employees accrue PTO in proportion to their scheduled hours. Up to 6 unused PTO days may be carried over into the following year; carried-over days expire on March 31. Contractors are not eligible for PTO.

Requests for more than 5 consecutive days must be submitted at least 3 weeks in advance through the PeopleHub portal. Your manager will respond within 5 business days.

2. Holidays
The company observes 11 paid holidays per year, plus a company-wide shutdown between December 24 and January 1. The holiday calendar for each year is published in PeopleHub every November.""",
    """3. Remote Work
Employees may work remotely up to 3 days per week with manager approval. Fully remote arrangements require approval from a department director and HR. Employees working remotely must be reachable during core hours of 10:00 to 15:00 in their local time zone.

The home office stipend is $600, paid once when you first set up a home office. An additional $40 per month is provided for internet costs.

4. Expenses
Expenses must be submitted within 30 days of being incurred. Any single expense above $250 requires a receipt and manager approval. Meals during business travel are reimbursed up to $75 per day. Economy class is required for flights under 6 hours; premium economy is permitted for longer flights. Alcohol is never reimbursed.""",
    """5. Equipment
New engineers receive a laptop with 64 GB of memory. Other new employees receive a laptop with 32 GB of memory. Laptops are refreshed every 3 years. Lost or stolen equipment must be reported to IT within 24 hours by filing a ticket in the ServiceDesk queue HW-LOSS.

6. Parental Leave
Primary caregivers receive 18 weeks of fully paid parental leave; secondary caregivers receive 8 weeks. Leave must begin within 12 months of the birth or adoption and may be split into at most two blocks.""",
    """7. Learning Budget
Each employee has an annual learning budget of $1,500 for courses, books, and conferences. Unused learning budget does not roll over. Conference travel counts against the learning budget only for the registration fee; travel itself is paid from the team travel budget.

8. Code of Conduct
All employees must complete the code of conduct training within 30 days of starting and every year after that. Concerns can be raised anonymously through the EthicsLine portal.""",
]


def build_handbook(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    for text in HANDBOOK_PAGES:
        page = doc.new_page()  # A4
        rc = page.insert_textbox(pymupdf.Rect(60, 60, 535, 780), text, fontsize=11, fontname="helv")
        if rc < 0:
            raise RuntimeError(f"Handbook page text overflowed by {-rc:.0f}pt")
    doc.save(path)


# ---------------------------------------------------------------- archived policy (MD)

ARCHIVED_PTO = """# PTO Policy (2024) - ARCHIVED

> **Archived.** This policy was superseded on January 1, 2026 by the Employee Handbook.
> It is kept for historical reference only.

## Accrual

Full-time employees accrue 18 days of paid time off per year. Up to 10 unused days
may be carried over and do not expire.

## Requests

Requests for more than 5 consecutive days must be submitted 2 weeks in advance by
emailing your manager.

## Home office

New remote employees received a one-time home office stipend of $400.
"""


# ---------------------------------------------------------------- project reports (DOCX)

def build_project_report(path: Path, title: str, paragraphs: list[str], budget: list[tuple[str, str, str]],
                         closing: list[str]) -> None:
    import docx

    d = docx.Document()
    d.add_heading(title, level=1)
    for p in paragraphs:
        d.add_paragraph(p)
    d.add_heading("Budget", level=2)
    table = d.add_table(rows=1, cols=3)
    for cell, text in zip(table.rows[0].cells, ["Category", "Budget", "Spent to date"], strict=True):
        cell.text = text
    for row in budget:
        cells = table.add_row().cells
        for cell, text in zip(cells, row, strict=True):
            cell.text = text
    d.add_heading("Milestones and risks", level=2)
    for p in closing:
        d.add_paragraph(p)
    d.save(path)


FALCON = dict(
    title="Project Falcon - Q3 2026 Status Report",
    paragraphs=[
        "Project Falcon is building the autonomous warehouse picking arm, model HR-9. "
        "The project lead is Dana Ortiz and the executive sponsor is Priya Raman.",
        "Overall status: Amber. The vision subsystem is two weeks behind schedule because of a "
        "delayed camera supplier (Lumetrix). Mechanical and control subsystems are on track.",
    ],
    budget=[
        ("Hardware", "$1,150,000", "$870,000"),
        ("Software", "$900,000", "$610,000"),
        ("Testing", "$350,000", "$95,000"),
        ("Total", "$2,400,000", "$1,575,000"),
    ],
    closing=[
        "Pilot deployment at the Reno warehouse is scheduled for February 9, 2027. "
        "General availability is targeted for June 2027.",
        "Top risk: supplier delays for the Lumetrix camera module. Mitigation: qualifying a second "
        "supplier, OptiCore, by November 30, 2026.",
    ],
)

OSPREY = dict(
    title="Project Osprey - Q3 2026 Status Report",
    paragraphs=[
        "Project Osprey is building the fleet management software for delivery drones. "
        "The project lead is Marcus Bell and the executive sponsor is Priya Raman.",
        "Overall status: Green. Route planning and telemetry are complete; the operator dashboard "
        "is in user testing.",
    ],
    budget=[
        ("Hardware", "$300,000", "$210,000"),
        ("Software", "$1,600,000", "$1,240,000"),
        ("Testing", "$250,000", "$140,000"),
        ("Total", "$2,150,000", "$1,590,000"),
    ],
    closing=[
        "A pilot with the city of Boise begins January 12, 2027.",
        "Top risk: FAA waiver approval for beyond-visual-line-of-sight flights. Mitigation: the "
        "application was filed in August and a consultant has been engaged.",
    ],
)


# ---------------------------------------------------------------- security runbook (MD)

RUNBOOK = """# IT Security Runbook

Use the incident code in the subject line of every report.

## SEC-417: Suspected phishing email

Do not click any links or open attachments. Forward the email as an attachment to
phish@halcyon-robotics.example and then delete it. The security team responds within
4 business hours.

## SEC-502: Lost or stolen device

Report the loss within 24 hours by filing a ticket in the HW-LOSS queue. IT will remotely
wipe the device. If the device contained customer data, also notify your manager.

## SEC-610: Production credentials exposed

Rotate the credentials immediately, then page the on-call security engineer via the
#sec-oncall channel. Severity is always P1. Do not delete logs.

## VPN error codes

- ERR_VPN_0x2F: certificate expired. Run `halcyon-vpn renew-cert` and reconnect.
- ERR_VPN_0x31: MFA timeout. Approve the push notification within 60 seconds.
- ERR_VPN_0x44: Account locked after 5 failed attempts. Contact the ServiceDesk to unlock it.

## Password policy

Passwords must be at least 14 characters long and are rotated every 180 days.
Multi-factor authentication (MFA) is mandatory for all accounts.
"""


# ---------------------------------------------------------------- meeting notes (HTML)

MEETINGS = """<!doctype html>
<html><head><title>Q3 2026 Meeting Notes</title></head>
<body>
<h1>Q3 2026 Meeting Notes</h1>

<h2>2026-07-14 Leadership sync</h2>
<p>Attendees: Priya Raman, Dana Ortiz, Marcus Bell, Leah Kim (CFO).</p>
<p>Decision: Priya Raman approved a $180,000 budget increase for Project Osprey to expand the testing program, on top of the budget in the Q3 status report.</p>
<p>Action item: Leah Kim will update the FY2027 forecast by July 31.</p>

<h2>2026-08-21 Engineering all-hands</h2>
<p>The team voted to standardize on Rust for all new firmware. Existing C++ firmware will be maintained but not rewritten.</p>
<p>The Reno warehouse pilot for Project Falcon was confirmed; the on-site lead will be Tomas Herrera.</p>

<h2>2026-09-30 Security review</h2>
<p>Incidents in Q3: 3 phishing reports (SEC-417), 1 lost laptop (SEC-502), 0 credential exposures (SEC-610).</p>
<p>Decision: the password rotation period will remain 180 days.</p>
</body></html>
"""


# ---------------------------------------------------------------- customer FAQ (TXT)

FAQ = """Halcyon Robotics Customer FAQ

Q: What is the warranty on Halcyon robotic arms?
A: All robotic arms include a 2-year limited warranty covering parts and labor. Extended warranties of up to 5 years can be purchased at checkout.

Q: How do I return a product?
A: Products can be returned within 45 days of delivery in their original packaging. A restocking fee of 15% applies to opened items.

Q: What are your support hours?
A: Phone support is available Monday to Friday, 7:00 to 19:00 Pacific Time. Emails to support@halcyon-robotics.example are answered within 1 business day.

Q: Do you ship internationally?
A: We ship to the United States, Canada, and the European Union. Customers elsewhere can order through an authorized reseller.

Q: How often should gripper pads be replaced?
A: Gripper pads should be replaced every 50,000 cycles or every 6 months, whichever comes first.
"""


# ---------------------------------------------------------------- generated CSVs (noise)

FIXED_SKUS = [
    ("HX-4471", "Servo Actuator S2", "Actuators", "149.00", "0.42", "Reno"),
    ("HX-5120", "Gripper Pad Set (10 pack)", "Grippers", "38.50", "0.15", "Boise"),
    ("HX-7781", "LiDAR Module L4", "Sensors", "1299.00", "0.95", "Reno"),
]

PRODUCT_WORDS = {
    "Actuators": ["Servo Actuator", "Linear Actuator", "Stepper Drive", "Harmonic Drive"],
    "Sensors": ["Proximity Sensor", "Torque Sensor", "Depth Camera", "IMU Board"],
    "Grippers": ["Vacuum Gripper", "Parallel Gripper", "Soft Gripper", "Gripper Finger Set"],
    "Controllers": ["Motion Controller", "Safety Relay", "I/O Expansion Board", "Edge Compute Unit"],
    "Cables": ["Power Cable", "EtherCAT Cable", "Encoder Cable", "Sensor Harness"],
    "Batteries": ["Battery Pack", "Charging Dock", "Battery Management Board", "Backup Cell"],
}

TICKET_SUBJECTS = [
    "Gripper pad wear question", "VPN will not connect", "Return request for opened item",
    "Arm calibration drift", "Invoice copy needed", "Firmware update failed", "Shipping delay",
    "Warranty claim for actuator", "Request for extended warranty", "Camera image is blurry",
    "Battery not charging", "Dashboard login issue", "Replacement cable request",
]
CUSTOMERS = ["Northgate Supply", "Bluewater Freight", "Kestrel Foods", "Summit Parts",
             "Orion Distribution", "Maple Logistics", "Redline Retail", "Tidewater Imports"]


def build_catalog() -> str:
    rng = random.Random(42)
    used = {s[0] for s in FIXED_SKUS} | {"HX-9999"}  # HX-9999 is used by an unanswerable question
    rows = list(FIXED_SKUS)
    while len(rows) < 500:
        sku = f"HX-{rng.randint(1000, 8999)}"
        if sku in used:
            continue
        used.add(sku)
        category = rng.choice(list(PRODUCT_WORDS))
        name = f"{rng.choice(PRODUCT_WORDS[category])} {rng.choice('ABCDEFGH')}{rng.randint(1, 9)}"
        rows.append((sku, name, category, f"{rng.uniform(5, 2500):.2f}", f"{rng.uniform(0.05, 6):.2f}",
                     rng.choice(["Reno", "Boise", "Denver"])))
    rng.shuffle(rows)
    return _csv(["sku", "name", "category", "price_usd", "weight_kg", "warehouse"], rows)


def build_tickets() -> str:
    rng = random.Random(7)
    rows = [("T-20931", "2026-09-14", "Acme Logistics", "Arm HR-9 stops mid-pick with error E-77", "escalated")]
    used = {"T-20931"}
    while len(rows) < 400:
        tid = f"T-{rng.randint(20000, 21999)}"
        if tid in used:
            continue
        used.add(tid)
        rows.append((tid, f"2026-{rng.randint(7, 9):02d}-{rng.randint(1, 28):02d}", rng.choice(CUSTOMERS),
                     rng.choice(TICKET_SUBJECTS), rng.choice(["open", "closed", "closed", "pending"])))
    rows.sort(key=lambda r: r[1])
    return _csv(["ticket_id", "date", "customer", "subject", "status"], rows)


DIGEST_PROJECTS = ["Heron", "Kestrel", "Wren", "Plover", "Sparrow"]
DIGEST_PEOPLE = ["Alicia Moreno", "Ben Okafor", "Chen Wei", "Divya Nair", "Erik Lund", "Fatima Zahra",
                 "Grace Park", "Hugo Silva", "Ines Duarte", "Jonas Weber", "Kofi Mensah", "Lena Novak"]
DIGEST_CITIES = ["Denver", "Tucson", "Spokane", "Fresno", "Omaha"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def build_digests() -> str:
    """Weekly digests about OTHER projects: semantically close to the real facts
    (budgets, pilots, leads, GA dates, statuses), but about different projects.
    These are the distractors that make retrieval genuinely hard."""
    rng = random.Random(11)

    def money(lo, hi):
        return f"${rng.randrange(lo, hi, 10_000):,}"

    def date():
        return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, 2027"

    templates = [
        lambda p: f"The {p} team moved its pilot with the city of {rng.choice(DIGEST_CITIES)} to {date()}.",
        lambda p: f"{p}'s {rng.choice(['hardware', 'software', 'testing'])} budget was revised to "
                  f"{money(100_000, 1_500_000)}, with {money(50_000, 900_000)} spent so far.",
        lambda p: f"{rng.choice(DIGEST_PEOPLE)} will lead the {p} {rng.choice(['vision', 'controls', 'firmware', 'cloud'])} "
                  f"workstream starting in {rng.choice(MONTHS)}.",
        lambda p: f"Project {p} reported status {rng.choice(['Green', 'Amber', 'Red'])} this week because of "
                  f"{rng.choice(['a late component shipment', 'staffing changes', 'a failed safety test', 'scope changes'])}.",
        lambda p: f"General availability for Project {p} is now targeted for {rng.choice(MONTHS)} 2027.",
        lambda p: f"The executive sponsor for Project {p} is {rng.choice(DIGEST_PEOPLE)}.",
        lambda p: f"Top risk for {p}: {rng.choice(['supplier lead times', 'regulatory review', 'battery certification', 'integration testing capacity'])}.",
        lambda p: f"The {p} team closed {rng.randint(3, 40)} support tickets about "
                  f"{rng.choice(['gripper wear', 'VPN access', 'calibration drift', 'firmware updates'])}.",
    ]
    weeks = []
    for week in range(20):
        lines = [f"## Week {week + 1} digest", ""]
        for _ in range(rng.randint(6, 9)):
            lines.append("- " + rng.choice(templates)(rng.choice(DIGEST_PROJECTS)))
        weeks.append("\n".join(lines))
    return "# Halcyon Robotics Weekly Digests (2026)\n\nShort updates from teams across the company.\n\n" + \
        "\n\n".join(weeks) + "\n"


def _csv(header: list[str], rows: list[tuple]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------- main

def main() -> None:
    CORPUS_DIR.mkdir(exist_ok=True)
    build_handbook(CORPUS_DIR / "employee_handbook.pdf")
    (CORPUS_DIR / "pto_policy_2024_archived.md").write_text(ARCHIVED_PTO)
    build_project_report(CORPUS_DIR / "project_falcon_status.docx", **FALCON)
    build_project_report(CORPUS_DIR / "project_osprey_status.docx", **OSPREY)
    (CORPUS_DIR / "it_security_runbook.md").write_text(RUNBOOK)
    (CORPUS_DIR / "meeting_notes_q3.html").write_text(MEETINGS)
    (CORPUS_DIR / "customer_faq.txt").write_text(FAQ)
    (CORPUS_DIR / "product_catalog.csv").write_text(build_catalog())
    (CORPUS_DIR / "support_tickets.csv").write_text(build_tickets())
    (CORPUS_DIR / "weekly_digests_2026.md").write_text(build_digests())
    for f in sorted(CORPUS_DIR.iterdir()):
        print(f"{f.name:32} {f.stat().st_size:>8,} bytes")


if __name__ == "__main__":
    main()
