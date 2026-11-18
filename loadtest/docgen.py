"""Generates a unique test document per simulated user (shared by locustfile and steps)."""

import random
import uuid


def make_document() -> str:
    """A ~6 KB policy document with per-user random facts (unique content per user)."""
    r = random.Random()
    facts = [
        f"Full-time employees receive {r.randint(15, 30)} PTO days per year. Part-time employees accrue PTO pro rata.",
        f"Remote employees get a one-time home office stipend of ${r.choice([400, 500, 600, 750])}.",
        f"Expense reports over ${r.choice([500, 1000, 2500])} must be approved by the department director.",
        f"Laptops are refreshed every {r.randint(2, 4)} years; contact IT to request an early replacement.",
        f"Parental leave is {r.choice([12, 16, 20])} weeks at full pay for all employees.",
        f"On-call engineers receive ${r.choice([200, 300, 400])} per on-call week plus time off in lieu.",
        "The required VPN client is WireGuard, configured through the IT self-service portal.",
        f"Performance reviews are held every {r.choice(['March and September', 'June and December'])}.",
    ]
    filler = (
        "This section describes general workplace expectations, communication norms, "
        "meeting etiquette and the escalation path for questions not covered elsewhere. "
    )
    paragraphs = []
    for i, fact in enumerate(facts, 1):
        paragraphs.append(f"## Section {i}\n\n{filler * 3}\n\n{fact}\n\n{filler * 2}")
    return f"# Employee Policy {uuid.uuid4().hex[:8]}\n\n" + "\n\n".join(paragraphs)
