"""Generate synthetic French accounting transaction data.

Books a single synthetic company's bank journal: transactions are allocated to account
codes, filled from templates, and written one row per journal line. The dataset is
deterministic for a given seed and contains zero proprietary data.

Usage:
    uv run python scripts/generate_sample_data.py
    uv run python scripts/generate_sample_data.py --classes 300 --rows 50000 --output data/generated/x.csv
"""

import argparse
import csv
import hashlib
import random
from datetime import date, timedelta
from functools import cache
from itertools import accumulate
from pathlib import Path
from typing import NamedTuple

DEFAULT_CLASSES = 100
DEFAULT_ROWS = 10_000
DEFAULT_SEED = 42
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample.csv"
DATE_START = date(2024, 1, 1)
DATE_END = date(2025, 12, 31)

# Every primary account gets at least this many transactions, so each class clears
# the loader's min_class_samples filter and reaches the training block.
MIN_TRANSACTIONS = 20

# Counterparty banks by country: BIC characters 5-6 carry the country code, and
# the int is the BBAN length that follows the IBAN's 4-character prefix.
BANKS_BY_COUNTRY: dict[str, tuple[list[str], int]] = {
    "FR": (["BNPAFRPP", "SOGEFRPP", "CRLYFRPP", "AGRIFRPP", "CCBPFRPP"], 23),
    "BE": (["GEBABEBB"], 12),
    "DE": (["DEUTDEFF"], 18),
}
# Mostly domestic counterparties, with a few foreign ones so is_domestic_iban varies.
COUNTRY_MIX = ["FR"] * 8 + ["BE", "DE"]

# ---------------------------------------------------------------------------
# Realism mechanisms (docs/DESIGN.md, "Synthetic data generator"). Each rule
# mirrors how real books behave; scripts/estimate_ceiling.py models every one.
# ---------------------------------------------------------------------------

# An account's counterparty is one of its few regular suppliers or clients most
# of the time, otherwise anyone in the account's pool.
REGULAR_COUNTERPARTIES = 3
REGULAR_COUNTERPARTY_RATE = 0.80

# Accounts paid on a calendar: (months or None for every month, first day,
# last day or None for the month's end). Other accounts use the default spread.
CALENDAR_RULES: dict[str, tuple[tuple[int, ...] | None, int, int | None]] = {
    # Social contributions: days 5-15.
    "431000": (None, 5, 15),
    "437000": (None, 5, 15),
    "645000": (None, 5, 15),
    "646000": (None, 5, 15),
    # VAT returns: days 17-24 of the month after each quarter.
    "445660": ((1, 4, 7, 10), 17, 24),
    "445710": ((1, 4, 7, 10), 17, 24),
    # Corporate tax instalments: days 10-20 of each quarter's last month.
    "691000": ((3, 6, 9, 12), 10, 20),
    # Salaries: from day 25 to the end of the month.
    "421000": (None, 25, None),
    "641000": (None, 25, None),
    "641100": (None, 25, None),
    # Rent: days 1-5.
    "613200": (None, 1, 5),
    # Year-end closing entries.
    "486000": ((12,), 20, None),
    "681000": ((12,), 20, None),
    "781000": ((12,), 20, None),
}

# Bank label channel: banks abbreviate the payment-type prefix differently, some
# labels lose a character, and every label is cut to the bank's display width.
# Structured remarks are machine-written and pass through untouched.
LABEL_VARIANTS: dict[str, list[tuple[str, float]]] = {
    "PRLV SEPA ": [("PRLV SEPA ", 0.60), ("PRELEVEMENT SEPA ", 0.25), ("PRLVT SEPA ", 0.15)],
    "VIR SEPA ": [("VIR SEPA ", 0.60), ("VIREMENT SEPA ", 0.25), ("VIR ", 0.15)],
    "VIR RECU ": [("VIR RECU ", 0.70), ("VIREMENT RECU ", 0.30)],
    "CB ": [("CB ", 0.60), ("CARTE ", 0.25), ("PAIEMENT CB ", 0.15)],
}
TYPO_RATE = 0.05
BANK_LABEL_LENGTH = 32

# Bookkeepers disagree between neighbouring accounts: this share of a paired
# account's rows is recorded under its sibling.
LABEL_NOISE_RATE = 0.03
SIBLING_ACCOUNTS: dict[str, str] = {
    code: sibling
    for pair in (
        ("606100", "606300"),
        ("625100", "625200"),
        ("641000", "641100"),
    )
    for code, sibling in (pair, pair[::-1])
}

# ---------------------------------------------------------------------------
# French PCG accounting codes with target counts (power-law distribution)
# ---------------------------------------------------------------------------
ACCOUNT_CODES: dict[str, int] = {
    # High frequency (370+ samples each)
    "601100": 432,
    "641000": 395,
    "512000": 370,
    # Medium frequency (99-222 each)
    "606100": 198,
    "613200": 185,
    "621000": 173,
    "625100": 160,
    "626000": 148,
    "627000": 136,
    "631000": 123,
    "635000": 117,
    "641100": 111,
    "645000": 105,
    "661000": 99,
    "706000": 222,
    "707000": 198,
    "708000": 173,
    # Low frequency (15-74 each) -- many codes
    "164000": 68,
    "205000": 62,
    "211000": 56,
    "213100": 49,
    "218000": 43,
    "261000": 37,
    "275000": 31,
    "403000": 62,
    "404000": 56,
    "408000": 49,
    "416000": 62,
    "419000": 56,
    "421000": 74,
    "431000": 68,
    "437000": 62,
    "445660": 56,
    "445710": 49,
    "467000": 43,
    "471000": 37,
    "486000": 31,
    "512100": 68,
    "511500": 68,
    "514000": 62,
    "530000": 56,
    "580000": 49,
    "602000": 43,
    "604000": 37,
    "606300": 31,
    "611000": 68,
    "612000": 62,
    "614000": 56,
    "615000": 49,
    "616000": 43,
    "618000": 37,
    "622000": 31,
    "623000": 68,
    "624000": 62,
    "625200": 56,
    "626100": 49,
    "628000": 43,
    "633000": 37,
    "637000": 31,
    "645100": 25,
    "646000": 25,
    "651000": 25,
    "658000": 19,
    "668000": 19,
    "671000": 19,
    "681000": 25,
    "691000": 19,
    "701000": 68,
    "704000": 62,
    "713000": 37,
    "741000": 31,
    "758000": 25,
    "761000": 19,
    "771000": 19,
    "781000": 19,
}

# ---------------------------------------------------------------------------
# Entity name pools
# ---------------------------------------------------------------------------
SUPPLIER_ENTITIES = [
    "SAS DUPONT",
    "SARL MARTIN",
    "EURL BERNARD",
    "SA GROUPE LEGRAND",
    "SAS MOREAU",
    "SARL PETIT",
    "SCI ROUX",
    "SA FOURNIER",
    "SAS LAMBERT",
    "SARL BONNET",
    "SAS GIRARD",
    "EURL MERCIER",
    "SA LEFEVRE",
    "SARL SIMON",
    "SAS LAURENT",
    "SAS FAURE",
    "SARL ANDRE",
    "SA PICARD",
    "SAS DUVAL",
    "SARL GAUTIER",
]

SERVICE_ENTITIES = [
    "ORANGE SA",
    "SFR",
    "BOUYGUES TELECOM",
    "FREE MOBILE",
    "OVH SAS",
    "EDF",
    "ENGIE",
    "VEOLIA",
    "AXA ASSURANCES",
    "ALLIANZ",
    "MAIF",
    "MACIF",
    "MICROSOFT",
    "GOOGLE CLOUD",
    "CABINET EXPERTISE COMPTABLE",
    "SARL CONSULTANT RH",
]

CLIENT_ENTITIES = [
    "SA GROUPE LEGRAND",
    "SAS TECHNOLOGY PLUS",
    "SARL COMMERCE DU SUD",
    "SA INDUSTRIE NORD",
    "SAS DIGITAL SERVICES",
    "EURL ARTISAN BOIS",
    "SA PHARMA SANTE",
    "SAS CONSEIL RH",
    "SARL TRANSPORT EXPRESS",
    "SA GROUPE ALIMENTAIRE",
    "SAS AGENCE MEDIA",
    "SA BTP CONSTRUCTIONS",
    "SAS ECO ENERGIE",
    "SARL MODE TEXTILE",
    "SA CHIMIE FRANCE",
    "SAS LOGISTIQUE PRO",
    "EURL DESIGN STUDIO",
    "SA AERO COMPOSANTS",
]

MONTHS_FR = [
    "JANVIER",
    "FEVRIER",
    "MARS",
    "AVRIL",
    "MAI",
    "JUIN",
    "JUILLET",
    "AOUT",
    "SEPTEMBRE",
    "OCTOBRE",
    "NOVEMBRE",
    "DECEMBRE",
]

QUARTERS_FR = ["T1", "T2", "T3", "T4"]

# ---------------------------------------------------------------------------
# Templates: (description_pattern, remarks_pattern | None, is_debit, (min_amount, max_amount))
# Each account code maps to a list of template variants.
# ---------------------------------------------------------------------------
TemplateSpec = tuple[str, str | None, bool, tuple[float, float]]

TEMPLATES: dict[str, list[TemplateSpec]] = {
    # --- 403-408: Other supplier accounts (debit) ---
    "403000": [
        ("VIR SEPA {entity}", "VIR SEPA NPY:{entity} LIB:EFFETS A PAYER", True, (1000, 30000)),
    ],
    "404000": [
        (
            "VIR SEPA {entity}",
            "VIR SEPA NPY:{entity} LIB:ACHAT IMMOBILISATION",
            True,
            (2000, 50000),
        ),
    ],
    "408000": [
        (
            "FACTURE NON PARVENUE {entity}",
            "FACTURE NON PARVENUE NBE:{entity} LIB:AVOIR A RECEVOIR",
            True,
            (100, 5000),
        ),
    ],
    # --- 416-419: Other customer accounts (credit) ---
    "416000": [
        ("VIR RECU {entity}", "VIR SEPA NPY:{entity} LIB:CREANCE DOUTEUSE", False, (100, 10000)),
    ],
    "419000": [
        ("VIR RECU {entity}", "VIR SEPA NPY:{entity} LIB:ACOMPTE CLIENT", False, (500, 20000)),
    ],
    # --- 421000: Personnel (debit) ---
    "421000": [
        ("VIR SEPA SALAIRE {month}", "VIR SEPA LIB:SALAIRE {month} {year}", True, (1300, 5000)),
    ],
    # --- 431xxx: URSSAF / Social charges (debit) ---
    "431000": [
        (
            "PRLV SEPA URSSAF",
            "PRLV SEPA CPY:{ref8} NBE:URSSAF LIB:COTISATIONS SOCIALES {quarter} {year}",
            True,
            (2000, 25000),
        ),
        ("PRLV SEPA URSSAF", "PRLV SEPA NBE:URSSAF LIB:COTISATIONS {month}", True, (1500, 15000)),
    ],
    "437000": [
        (
            "PRLV SEPA POLE EMPLOI",
            "PRLV SEPA NBE:FRANCE TRAVAIL LIB:COTISATIONS CHOMAGE {quarter}",
            True,
            (500, 8000),
        ),
    ],
    # --- 445xxx: Tax (TVA / IS) ---
    "445660": [
        (
            "VIR SEPA DGFIP TVA",
            "VIR SEPA NPY:DGFIP LIB:TVA DEDUCTIBLE {quarter} {year}",
            True,
            (1000, 20000),
        ),
    ],
    "445710": [
        (
            "VIR RECU DGFIP",
            "VIR SEPA NPY:DGFIP LIB:REMBOURSEMENT TVA {quarter} {year}",
            False,
            (500, 30000),
        ),
        (
            "VIR SEPA DGFIP TVA",
            "VIR SEPA NPY:DGFIP LIB:TVA COLLECTEE {quarter}",
            True,
            (1000, 25000),
        ),
    ],
    # --- 467000: Sundry debtors/creditors ---
    "467000": [
        ("VIR SEPA {entity}", "VIR SEPA NPY:{entity} LIB:COMPTE TRANSITOIRE", True, (100, 10000)),
    ],
    "471000": [
        ("OPERATION ATTENTE", "LIB:OPERATION EN ATTENTE REF:{ref8}", True, (50, 5000)),
    ],
    "486000": [
        (
            "CHARGE CONSTATEE AVANCE",
            "LIB:CHARGE CONSTATEE D AVANCE {month} {year}",
            True,
            (200, 8000),
        ),
    ],
    # --- 512xxx: Bank transactions (mix) ---
    "512000": [
        (
            "VIR INTERNE",
            "VIR INTERNE REF:{ref8} LIB:VIREMENT DE COMPTE A COMPTE",
            True,
            (500, 50000),
        ),
        ("FRAIS BANCAIRES", "FRAIS BANCAIRES LIB:FRAIS DE GESTION {month}", True, (5, 200)),
        ("REMISE CHEQUE", "REMISE CHQ N {ref6}", False, (100, 20000)),
        ("VIR RECU {entity}", "VIR SEPA NPY:{entity} LIB:VIREMENT RECU", False, (200, 30000)),
    ],
    "512100": [
        ("VIR INTERNE", "VIR INTERNE REF:{ref8} LIB:VIREMENT ENTRE COMPTES", True, (1000, 100000)),
    ],
    "514000": [
        ("REMISE CHEQUE", "REMISE CHQ N {ref6} LIB:ENCAISSEMENT CHEQUE", False, (50, 10000)),
        ("REMISE CHEQUE", "REMISE CHQ N {ref6}", False, (100, 15000)),
    ],
    # --- 5115: Card remittances ---
    "511500": [
        ("ENCAISSEMENT CB", "REMISE CB DU {date6} LIB:ENCAISSEMENT CARTE", False, (15, 3000)),
    ],
    "530000": [
        ("RETRAIT CAISSE", "RETRAIT DAB LIB:APPROVISIONNEMENT CAISSE", True, (50, 1000)),
    ],
    "580000": [
        ("VIR INTERNE", "VIR INTERNE LIB:VIREMENT INTERNE", True, (500, 50000)),
    ],
    # --- 601xxx: Purchases (debit) ---
    "601100": [
        ("CB {entity}", "CB {entity} FACT:{inv} LIB:ACHAT FOURNITURES", True, (10, 2000)),
        ("CB {entity}", "CB {entity} LIB:ACHAT MARCHANDISES", True, (20, 5000)),
        ("PRLV SEPA {entity}", "PRLV SEPA NBE:{entity} LIB:COMMANDE {inv}", True, (50, 8000)),
    ],
    "602000": [
        ("CB {entity}", "CB {entity} LIB:ACHAT MATIERES PREMIERES", True, (100, 15000)),
    ],
    "604000": [
        ("CB {entity}", "CB {entity} LIB:ACHAT ETUDES PRESTATIONS", True, (200, 10000)),
    ],
    "606100": [
        ("CB {entity}", "CB {entity} LIB:FOURNITURES NON STOCKABLES", True, (5, 500)),
        ("PRLV SEPA {entity}", "PRLV SEPA NBE:{entity} LIB:EAU ELECTRICITE", True, (30, 800)),
    ],
    "606300": [
        ("CB {entity}", "CB {entity} LIB:PETIT EQUIPEMENT", True, (10, 2000)),
    ],
    # --- 611-618: Services exterieurs ---
    "611000": [
        ("PRLV SEPA {entity}", "PRLV SEPA NBE:{entity} LIB:SOUS-TRAITANCE", True, (500, 20000)),
    ],
    "612000": [
        ("PRLV SEPA {entity}", "PRLV SEPA NBE:{entity} LIB:CREDIT-BAIL", True, (200, 5000)),
    ],
    "613200": [
        (
            "PRLV SEPA {entity}",
            "PRLV SEPA NBE:{entity} LIB:LOYER {month} {year}",
            True,
            (500, 5000),
        ),
    ],
    "614000": [
        (
            "PRLV SEPA {entity}",
            "PRLV SEPA NBE:{entity} LIB:CHARGES LOCATIVES {quarter}",
            True,
            (100, 2000),
        ),
    ],
    "615000": [
        (
            "PRLV SEPA {entity}",
            "PRLV SEPA NBE:{entity} LIB:ENTRETIEN REPARATIONS",
            True,
            (50, 3000),
        ),
    ],
    "616000": [
        ("PRLV SEPA {entity}", "PRLV SEPA NBE:{entity} LIB:ASSURANCE {year}", True, (200, 5000)),
        (
            "PRLV SEPA AXA ASSURANCES",
            "PRLV SEPA NBE:AXA ASSURANCES LIB:PRIME ASSURANCE",
            True,
            (100, 3000),
        ),
    ],
    "618000": [
        ("CB {entity}", "CB {entity} LIB:DOCUMENTATION TECHNIQUE", True, (20, 500)),
    ],
    # --- 621-628: Autres services exterieurs ---
    "621000": [
        (
            "PRLV SEPA {entity}",
            "PRLV SEPA NBE:{entity} LIB:PERSONNEL INTERIMAIRE",
            True,
            (500, 10000),
        ),
    ],
    "622000": [
        (
            "VIR SEPA {entity}",
            "VIR SEPA NPY:{entity} LIB:HONORAIRES COMPTABLES",
            True,
            (500, 8000),
        ),
    ],
    "623000": [
        ("CB {entity}", "CB {entity} LIB:PUBLICITE ANNONCES", True, (50, 5000)),
    ],
    "624000": [
        ("CB {entity}", "CB {entity} LIB:TRANSPORT LIVRAISON", True, (10, 500)),
    ],
    "625100": [
        ("CB {entity}", "CB {entity} LIB:DEPLACEMENT MISSION", True, (20, 1000)),
        ("CB {entity}", "CB {entity} LIB:FRAIS VOYAGE", True, (50, 2000)),
    ],
    "625200": [
        ("CB {entity}", "CB {entity} LIB:FRAIS HEBERGEMENT", True, (50, 500)),
    ],
    "626000": [
        ("PRLV SEPA LA POSTE", "PRLV SEPA NBE:LA POSTE LIB:AFFRANCHISSEMENT", True, (5, 200)),
        ("PRLV SEPA {entity}", "PRLV SEPA NBE:{entity} LIB:TELECOMMUNICATIONS", True, (20, 300)),
    ],
    "626100": [
        ("PRLV SEPA {entity}", "PRLV SEPA NBE:{entity} LIB:FRAIS TELEPHONIQUES", True, (15, 200)),
    ],
    "627000": [
        ("PRLV SEPA ORANGE SA", "PRLV SEPA NBE:ORANGE SA LIB:ABONNEMENT MOBILE", True, (20, 100)),
        ("PRLV SEPA SFR", "PRLV SEPA NBE:SFR LIB:FORFAIT MOBILE", True, (15, 80)),
        ("PRLV SEPA OVH SAS", "PRLV SEPA NBE:OVH SAS LIB:HEBERGEMENT SERVEUR", True, (10, 500)),
        (
            "PRLV SEPA BOUYGUES TELECOM",
            "PRLV SEPA NBE:BOUYGUES TELECOM LIB:ABONNEMENT INTERNET",
            True,
            (25, 100),
        ),
    ],
    "628000": [
        ("CB {entity}", "CB {entity} LIB:SERVICES BANCAIRES DIVERS", True, (5, 200)),
    ],
    # --- 631-637: Impots et taxes ---
    "631000": [
        (
            "PRLV SEPA DGFIP",
            "PRLV SEPA NBE:DGFIP LIB:TAXE APPRENTISSAGE {year}",
            True,
            (500, 5000),
        ),
        (
            "PRLV SEPA DGFIP",
            "PRLV SEPA NBE:DGFIP LIB:FORMATION CONTINUE {year}",
            True,
            (200, 3000),
        ),
    ],
    "633000": [
        (
            "PRLV SEPA DGFIP",
            "PRLV SEPA NBE:DGFIP LIB:CONTRIBUTION ECONOMIQUE TERRITORIALE",
            True,
            (300, 8000),
        ),
    ],
    "635000": [
        ("PRLV SEPA DGFIP", "PRLV SEPA NBE:DGFIP LIB:AUTRES IMPOTS {year}", True, (100, 5000)),
    ],
    "637000": [
        (
            "PRLV SEPA DGFIP",
            "PRLV SEPA NBE:DGFIP LIB:TAXE SUR SALAIRES {quarter}",
            True,
            (200, 3000),
        ),
    ],
    # --- 641xxx: Salaries (debit) ---
    "641000": [
        ("VIR SEPA SALAIRE {month}", "VIR SEPA LIB:SALAIRE {month} {year}", True, (1300, 5000)),
        ("VIR SEPA SALAIRE {month}", "VIR SEPA LIB:REMUNERATION {month}", True, (1800, 4500)),
    ],
    "641100": [
        (
            "VIR SEPA SALAIRE {month}",
            "VIR SEPA LIB:CONGES PAYES {month} {year}",
            True,
            (1300, 4000),
        ),
    ],
    # --- 645xxx: Social charges (debit) ---
    "645000": [
        (
            "PRLV SEPA URSSAF",
            "PRLV SEPA NBE:URSSAF LIB:COTISATIONS PATRONALES {quarter} {year}",
            True,
            (2000, 20000),
        ),
    ],
    "645100": [
        (
            "PRLV SEPA MUTUELLE",
            "PRLV SEPA NBE:MUTUELLE LIB:PREVOYANCE SANTE {month}",
            True,
            (200, 2000),
        ),
    ],
    "646000": [
        (
            "PRLV SEPA POLE EMPLOI",
            "PRLV SEPA NBE:POLE EMPLOI LIB:COTISATIONS CHOMAGE {quarter}",
            True,
            (500, 5000),
        ),
    ],
    # --- 651-658: Other charges ---
    "651000": [
        ("VIR SEPA {entity}", "VIR SEPA NPY:{entity} LIB:REDEVANCE BREVET", True, (100, 5000)),
    ],
    "658000": [
        (
            "VIR SEPA {entity}",
            "VIR SEPA NPY:{entity} LIB:CHARGES GESTION COURANTE",
            True,
            (50, 2000),
        ),
    ],
    # --- 661xxx: Financial charges (debit) ---
    "661000": [
        (
            "AGIOS BANCAIRES",
            "AGIOS BANCAIRES LIB:AGIOS TRIMESTRIELS {quarter} {year}",
            True,
            (10, 500),
        ),
        ("INTERETS EMPRUNT", "INTERETS LIB:INTERET DEBITEUR {month} {year}", True, (50, 2000)),
        ("FRAIS BANCAIRES", "FRAIS BANCAIRES LIB:COMMISSION BANCAIRE {month}", True, (5, 150)),
    ],
    "668000": [
        ("FRAIS FINANCIERS", "LIB:CHARGES FINANCIERES DIVERSES", True, (10, 500)),
    ],
    "671000": [
        (
            "VIR SEPA {entity}",
            "VIR SEPA NPY:{entity} LIB:CHARGE EXCEPTIONNELLE",
            True,
            (100, 10000),
        ),
    ],
    "681000": [
        (
            "DOTATION AMORTISSEMENT",
            "LIB:DOTATION AMORTISSEMENT {quarter} {year}",
            True,
            (500, 15000),
        ),
    ],
    "691000": [
        (
            "VIR SEPA DGFIP IS",
            "VIR SEPA NPY:DGFIP LIB:IMPOT SUR SOCIETES {year}",
            True,
            (1000, 50000),
        ),
    ],
    # --- 164000: Long-term debt ---
    "164000": [
        (
            "PRLV SEPA ECHEANCE EMPRUNT",
            "PRLV SEPA LIB:ECHEANCE EMPRUNT {ref8}",
            True,
            (500, 10000),
        ),
    ],
    # --- 2xxxxx: Immobilisations ---
    "205000": [
        ("VIR SEPA {entity}", "VIR SEPA NPY:{entity} LIB:LOGICIEL {inv}", True, (1000, 30000)),
    ],
    "211000": [
        ("VIR SEPA {entity}", "VIR SEPA NPY:{entity} LIB:TERRAIN", True, (10000, 200000)),
    ],
    "213100": [
        ("VIR SEPA {entity}", "VIR SEPA NPY:{entity} LIB:CONSTRUCTION", True, (5000, 150000)),
    ],
    "218000": [
        (
            "VIR SEPA {entity}",
            "VIR SEPA NPY:{entity} LIB:IMMOBILISATION CORPORELLE",
            True,
            (1000, 50000),
        ),
    ],
    "261000": [
        (
            "VIR SEPA {entity}",
            "VIR SEPA NPY:{entity} LIB:TITRES DE PARTICIPATION",
            True,
            (5000, 100000),
        ),
    ],
    "275000": [
        ("VIR SEPA {entity}", "VIR SEPA NPY:{entity} LIB:DEPOT GARANTIE", True, (500, 10000)),
    ],
    # --- 70xxxx: Revenue (credit) ---
    "701000": [
        (
            "VIR RECU {entity}",
            "VIR SEPA NPY:{entity} LIB:VENTE PRODUITS FINIS {inv}",
            False,
            (200, 30000),
        ),
    ],
    "704000": [
        ("VIR RECU {entity}", "VIR SEPA NPY:{entity} LIB:TRAVAUX {inv}", False, (500, 50000)),
    ],
    "706000": [
        (
            "VIR RECU {entity}",
            "VIR SEPA NPY:{entity} LIB:PRESTATIONS SERVICES {inv}",
            False,
            (100, 20000),
        ),
        ("VIR RECU {entity}", "VIR SEPA NPY:{entity} LIB:HONORAIRES {month}", False, (500, 15000)),
    ],
    "707000": [
        (
            "VIR RECU {entity}",
            "VIR SEPA NPY:{entity} LIB:VENTE MARCHANDISES {inv}",
            False,
            (50, 25000),
        ),
    ],
    "708000": [
        (
            "VIR RECU {entity}",
            "VIR SEPA NPY:{entity} LIB:PRODUITS ACTIVITES ANNEXES",
            False,
            (50, 5000),
        ),
    ],
    "713000": [
        ("VIR RECU {entity}", "VIR SEPA NPY:{entity} LIB:VARIATION STOCKS", False, (100, 10000)),
    ],
    "741000": [
        (
            "VIR RECU SUBVENTION",
            "VIR SEPA NPY:{entity} LIB:SUBVENTION EXPLOITATION {year}",
            False,
            (1000, 50000),
        ),
    ],
    "758000": [
        (
            "VIR RECU {entity}",
            "VIR SEPA NPY:{entity} LIB:PRODUITS GESTION COURANTE",
            False,
            (50, 3000),
        ),
    ],
    "761000": [
        (
            "VIR RECU INTERETS",
            "VIR SEPA LIB:INTERET CREDITEUR {quarter} {year}",
            False,
            (10, 1000),
        ),
    ],
    "771000": [
        (
            "VIR RECU {entity}",
            "VIR SEPA NPY:{entity} LIB:PRODUIT EXCEPTIONNEL",
            False,
            (100, 20000),
        ),
    ],
    "781000": [
        (
            "REPRISE AMORTISSEMENT",
            "LIB:REPRISE AMORTISSEMENT {quarter} {year}",
            False,
            (200, 10000),
        ),
    ],
}

# Accounts that share wording and differ only by amount: the same equipment
# purchase is expensed below the EUR 500 capitalisation threshold and capitalised
# from it, and a loan instalment splits into capital (164000) and interest.
AMOUNT_DECIDED_TEMPLATES: dict[str, list[TemplateSpec]] = {
    "606300": [("CB {entity}", "CB {entity} LIB:ACHAT MATERIEL", True, (10, 499.99))],
    "218000": [("CB {entity}", "CB {entity} LIB:ACHAT MATERIEL", True, (500, 15000))],
    "661000": [
        ("PRLV SEPA ECHEANCE EMPRUNT", "PRLV SEPA LIB:ECHEANCE EMPRUNT {ref8}", True, (20, 800))
    ],
}
for _code, _variants in AMOUNT_DECIDED_TEMPLATES.items():
    TEMPLATES[_code].extend(_variants)


# ---------------------------------------------------------------------------
# Counterparty sub-accounts (PCG subdivisions of 401 suppliers and 411 customers)
# ---------------------------------------------------------------------------

SUB_ACCOUNT_PREFIXES = ("401", "411")
MAX_SUB_ACCOUNTS = 999
# Combined weight of the generic codes the sub-accounts replace: 401000, 401100 and
# 401200 for suppliers; 411000 and 411100 for customers (411200's goes to 511500).
SUPPLIER_WEIGHT = 846
CUSTOMER_WEIGHT = 704

SUPPLIER_TEMPLATES: list[TemplateSpec] = [
    (
        "PRLV SEPA {entity}",
        "PRLV SEPA CPY:{ref8} RUM:{ref8} NBE:{entity} LIB:REGLEMENT FACTURE",
        True,
        (100, 15000),
    ),
    (
        "VIR SEPA {entity}",
        "VIR SEPA REF:{ref8} NPY:{entity} LIB:COMMANDE FOURNITURES LCC:BON COMMANDE {ref6}",
        True,
        (200, 25000),
    ),
    (
        "PRLV SEPA {entity}",
        "PRLV SEPA CPY:{ref8} NBE:{entity} LIB:FACTURE {inv}",
        True,
        (50, 8000),
    ),
    (
        "VIR SEPA {entity}",
        "VIR SEPA REF:{ref8} NPY:{entity} LIB:FACTURE FOURNISSEUR {inv} LC2:ECHEANCE {month}",
        True,
        (150, 12000),
    ),
    ("PRLV SEPA {entity}", "PRLV SEPA NBE:{entity} LIB:REGLEMENT ACHAT", True, (80, 5000)),
    (
        "VIR SEPA {entity}",
        "VIR SEPA REF:{ref8} NPY:{entity} LIB:ACOMPTE FOURNISSEUR",
        True,
        (500, 20000),
    ),
]

CUSTOMER_TEMPLATES: list[TemplateSpec] = [
    (
        "VIR RECU {entity}",
        "VIR SEPA NPY:{entity} IBE:{iban} BIC:{bic} PDO:{country} "
        "RCN:{inv} LIB:REGLEMENT FACTURE {inv}",
        False,
        (500, 50000),
    ),
    (
        "VIR RECU {entity}",
        "VIR SEPA NPY:{entity} IBE:{iban} BIC:{bic} PDO:{country} LIB:REGLEMENT CLIENT {inv}",
        False,
        (200, 30000),
    ),
]

SURNAMES = [
    "MARTIN", "BERNARD", "THOMAS", "PETIT", "ROBERT", "RICHARD", "DURAND", "DUBOIS",
    "MOREAU", "LAURENT", "SIMON", "MICHEL", "LEFEBVRE", "LEROY", "ROUX", "DAVID",
    "BERTRAND", "MOREL", "FOURNIER", "GIRARD", "BONNET", "DUPONT", "LAMBERT", "FONTAINE",
    "ROUSSEAU", "VINCENT", "MULLER", "LEFEVRE", "FAURE", "ANDRE", "MERCIER", "BLANC",
    "GUERIN", "BOYER", "GARNIER", "CHEVALIER", "FRANCOIS", "LEGRAND", "GAUTHIER", "GARCIA",
    "PERRIN", "ROBIN", "CLEMENT", "MORIN", "NICOLAS", "HENRY", "ROUSSEL", "MATHIEU",
    "GAUTIER", "MASSON", "MARCHAND", "DUVAL", "DENIS", "DUMONT", "MARIE", "LEMAIRE",
    "NOEL", "MEYER", "DUFOUR", "MEUNIER",
]  # fmt: skip
ACTIVITIES = [
    "BTP", "TRANSPORTS", "CONSEIL", "NEGOCE", "INDUSTRIE", "SERVICES", "DISTRIBUTION",
    "IMMOBILIER", "INFORMATIQUE", "LOGISTIQUE", "RESTAURATION", "SECURITE", "NETTOYAGE",
    "IMPRIMERIE", "ELECTRICITE",
]  # fmt: skip
LEGAL_FORMS = ["SAS", "SARL", "EURL", "SA", "SCI"]


@cache
def counterparty_names() -> list[str]:
    """Every sub-account counterparty name, in a fixed hash order."""
    names = [
        f"{form} {surname} {activity}"
        for form in LEGAL_FORMS
        for surname in SURNAMES
        for activity in ACTIVITIES
    ]
    return sorted(names, key=lambda name: hashlib.sha256(name.encode()).hexdigest())


def is_sub_account(account_code: str) -> bool:
    return account_code[:3] in SUB_ACCOUNT_PREFIXES


def sub_account_name(account_code: str) -> str:
    """The single counterparty booked to a supplier or customer sub-account.

    Suppliers take the even positions of counterparty_names() and customers the odd
    ones, so no name is both.
    """
    index = int(account_code[3:]) - 1
    return counterparty_names()[2 * index + SUB_ACCOUNT_PREFIXES.index(account_code[:3])]


def sub_account_codes(n_classes: int) -> list[str]:
    """Supplier then customer sub-accounts filling the classes the general catalogue leaves."""
    n_sub_accounts = n_classes - len(general_codes())
    if n_sub_accounts < 2:
        raise ValueError(f"--classes must be at least {len(general_codes()) + 2}, got {n_classes}")
    n_suppliers, n_customers = (n_sub_accounts + 1) // 2, n_sub_accounts // 2
    if n_suppliers > MAX_SUB_ACCOUNTS:
        raise ValueError(f"--classes allows at most {MAX_SUB_ACCOUNTS} sub-accounts per prefix")
    return [f"401{i:03d}" for i in range(1, n_suppliers + 1)] + [
        f"411{i:03d}" for i in range(1, n_customers + 1)
    ]


@cache
def general_codes() -> frozenset[str]:
    """Every account code the general catalogue can put in the output."""
    return frozenset(ACCOUNT_CODES)


def lines_per_transaction(account_code: str) -> int:
    """Journal lines one transaction of this primary account books."""
    return 1


def transaction_floor(account_code: str) -> int:
    """Minimum transactions for a primary account."""
    return MIN_TRANSACTIONS


def transaction_counts(n_classes: int, n_rows: int) -> dict[str, int]:
    """Transactions per primary account, so the output has about *n_rows* lines.

    General accounts keep their catalogue weights; sub-accounts share the supplier and
    customer weights by a Zipf law with exponent 1. Every account gets its floor, and one
    scale factor, found by bisection, sizes the rest.
    """
    weights: dict[str, float] = dict(ACCOUNT_CODES)
    sub_accounts = sub_account_codes(n_classes)
    for prefix, total in (("401", SUPPLIER_WEIGHT), ("411", CUSTOMER_WEIGHT)):
        family = [code for code in sub_accounts if code.startswith(prefix)]
        harmonic = sum(1 / rank for rank in range(1, len(family) + 1))
        for rank, code in enumerate(family, start=1):
            weights[code] = total / (rank * harmonic)

    def counts(scale: float) -> dict[str, int]:
        return {
            code: max(transaction_floor(code), round(w * scale)) for code, w in weights.items()
        }

    def total_lines(scale: float) -> int:
        return sum(n * lines_per_transaction(code) for code, n in counts(scale).items())

    minimum = total_lines(0.0)
    if minimum > n_rows:
        raise ValueError(f"--rows must be at least {minimum} for {n_classes} classes")
    low, high = 0.0, 1.0
    while total_lines(high) < n_rows:
        high *= 2
    for _ in range(60):
        mid = (low + high) / 2
        if total_lines(mid) < n_rows:
            low = mid
        else:
            high = mid
    return counts(high)


def templates_for(account_code: str) -> list[TemplateSpec]:
    if account_code.startswith("401"):
        return SUPPLIER_TEMPLATES
    if account_code.startswith("411"):
        return CUSTOMER_TEMPLATES
    return TEMPLATES[account_code]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_ALL_DAYS = [
    DATE_START + timedelta(days=offset) for offset in range((DATE_END - DATE_START).days + 1)
]


def _default_date_distribution() -> dict[date, float]:
    """Dates for accounts without a calendar rule.

    40% cluster at month end (a uniform year, month and day 25-31, clamped to the
    month's last day) to mimic treasury patterns; the rest spread uniformly.
    """
    pmf = dict.fromkeys(_ALL_DAYS, 0.6 / len(_ALL_DAYS))
    for year in (2024, 2025):
        for month in range(1, 13):
            next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
            last_day = (next_month - timedelta(days=1)).day
            for day in range(25, 32):
                pmf[date(year, month, min(day, last_day))] += 0.4 / (2 * 12 * 7)
    return pmf


@cache
def date_distribution(account_code: str) -> dict[date, float]:
    """P(posting date | account code), read as exact probabilities by estimate_ceiling.py."""
    rule = CALENDAR_RULES.get(account_code)
    if rule is None:
        return _default_date_distribution()
    months, first_day, last_day = rule
    days = [
        day
        for day in _ALL_DAYS
        if (months is None or day.month in months)
        and first_day <= day.day
        and (last_day is None or day.day <= last_day)
    ]
    return dict.fromkeys(days, 1 / len(days))


@cache
def _date_sampler(account_code: str) -> tuple[list[date], list[float]]:
    pmf = date_distribution(account_code)
    return list(pmf), list(accumulate(pmf.values()))


def pick_date(rng: random.Random, account_code: str) -> date:
    """Draw a posting date from the account's date distribution."""
    days, cum_weights = _date_sampler(account_code)
    return rng.choices(days, cum_weights=cum_weights)[0]


def counterparty_bank(entity: str) -> tuple[str, str, str]:
    """Return a stable (IBAN, BIC, country) for a counterparty name.

    Derived from a hash of the name rather than the seeded RNG: a counterparty
    keeps one account across rows, and filling these placeholders never shifts
    the random stream that every other generated value depends on.
    """
    digest = int(hashlib.sha256(entity.encode()).hexdigest(), 16)
    country = COUNTRY_MIX[digest % len(COUNTRY_MIX)]
    bics, bban_length = BANKS_BY_COUNTRY[country]
    bic = bics[(digest // len(COUNTRY_MIX)) % len(bics)]
    check_digits = 10 + digest % 90
    bban = digest % 10**bban_length
    return f"{country}{check_digits}{bban:0{bban_length}d}", bic, country


def fill_template(rng: random.Random, template: str, entity: str, tx_date: date) -> str:
    """Replace placeholders in a template string."""
    iban, bic, country = counterparty_bank(entity)
    result = template
    result = result.replace("{entity}", entity)
    result = result.replace("{iban}", iban)
    result = result.replace("{bic}", bic)
    result = result.replace("{country}", country)
    result = result.replace("{ref8}", f"FR{rng.randint(10000000, 99999999)}")
    result = result.replace("{ref6}", str(rng.randint(100000, 999999)))
    result = result.replace("{inv}", f"FAC{rng.randint(2024000, 2025999)}")
    result = result.replace("{date6}", tx_date.strftime("%d%m%y"))
    result = result.replace("{month}", MONTHS_FR[tx_date.month - 1])
    result = result.replace("{quarter}", QUARTERS_FR[(tx_date.month - 1) // 3])
    result = result.replace("{year}", str(tx_date.year))
    return result


# Share of amounts drawn as a round multiple, and the multiples used.
ROUND_AMOUNT_RATE = 0.2
ROUND_MAGNITUDES = [10, 50, 100, 500, 1000]
# Share of transactions that carry structured remarks.
REMARKS_RATE = 0.30


def generate_amount_cents(rng: random.Random, low: float, high: float) -> int:
    """Draw an amount in cents: a round multiple clipped to the range, or a uniform cent value."""
    low_cents, high_cents = round(low * 100), round(high * 100)
    if rng.random() < ROUND_AMOUNT_RATE:
        magnitude = rng.choice(ROUND_MAGNITUDES)
        multiple = magnitude * rng.randint(1, max(1, int(high / magnitude))) * 100
        return min(high_cents, max(low_cents, multiple))
    return rng.randint(low_cents, high_cents)


def format_comment_html(comment: str) -> str:
    """Format comment with HTML <br /> tags, splitting on LIB: markers."""
    if " LIB:" in comment:
        parts = comment.split(" LIB:", 1)
        return f"{parts[0]}<br />LIB:{parts[1]}"
    return comment


def entity_pool(account_code: str) -> list[str]:
    """Counterparty names a row of this account code can carry."""
    if is_sub_account(account_code):
        return [sub_account_name(account_code)]
    prefix = account_code[:3]
    if prefix in (
        "416",
        "419",
        "701",
        "704",
        "706",
        "707",
        "708",
        "713",
        "741",
        "758",
        "771",
    ):
        return CLIENT_ENTITIES
    elif prefix in ("627",):
        return SERVICE_ENTITIES[:8]  # Telecom/utilities
    elif prefix in ("616",):
        return SERVICE_ENTITIES[8:12]  # Insurance
    elif prefix in ("601", "606", "604", "618", "623", "624", "625", "628"):
        return SUPPLIER_ENTITIES + SERVICE_ENTITIES
    else:
        return SUPPLIER_ENTITIES


@cache
def entity_weights(account_code: str) -> dict[str, float]:
    """P(counterparty | account code): mostly a few regulars, otherwise anyone in the pool."""
    names = list(dict.fromkeys(entity_pool(account_code)))
    regulars = sorted(
        names, key=lambda name: hashlib.sha256(f"{account_code}:{name}".encode()).hexdigest()
    )[:REGULAR_COUNTERPARTIES]
    weights = dict.fromkeys(names, (1 - REGULAR_COUNTERPARTY_RATE) / len(names))
    for name in regulars:
        weights[name] += REGULAR_COUNTERPARTY_RATE / len(regulars)
    return weights


def pick_entity(rng: random.Random, account_code: str) -> str:
    """Draw a counterparty from the account's counterparty distribution."""
    weights = entity_weights(account_code)
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def label_variants(description_pattern: str) -> list[tuple[str, float]]:
    """How banks render a description pattern's payment-type prefix, with probabilities."""
    for prefix, variants in LABEL_VARIANTS.items():
        if description_pattern.startswith(prefix):
            rest = description_pattern[len(prefix) :]
            return [(rendered + rest, p) for rendered, p in variants]
    return [(description_pattern, 1.0)]


def bank_label(rng: random.Random, description: str) -> str:
    """Pass a filled description through the bank: a dropped character, then truncation."""
    if rng.random() < TYPO_RATE:
        position = rng.randrange(len(description))
        description = description[:position] + description[position + 1 :]
    return description[:BANK_LABEL_LENGTH]


def recorded_account(rng: random.Random, account_code: str) -> str:
    """The account a bookkeeper records: occasionally the sibling account."""
    sibling = SIBLING_ACCOUNTS.get(account_code)
    if sibling is not None and rng.random() < LABEL_NOISE_RATE:
        return sibling
    return account_code


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class Line(NamedTuple):
    """One journal line of a bank transaction, the unit the CSV stores as a row."""

    account_code: str
    cents: int
    is_debit: bool


class Transaction(NamedTuple):
    primary: str
    cents: int
    posting_date: date
    description: str
    remarks: str
    reference: str
    lines: tuple[Line, ...]


def generate(n_classes: int, n_rows: int, seed: int) -> list[Transaction]:
    """Book a company's bank journal with exactly *n_classes* account codes."""
    rng = random.Random(seed)
    transactions: list[Transaction] = []
    for primary, count in transaction_counts(n_classes, n_rows).items():
        templates = templates_for(primary)
        for _ in range(count):
            description_pattern, remarks_pattern, is_debit, (low, high) = rng.choice(templates)
            entity = pick_entity(rng, primary)
            tx_date = pick_date(rng, primary)
            cents = generate_amount_cents(rng, low, high)

            variants = label_variants(description_pattern)
            label_pattern = rng.choices(
                [rendered for rendered, _ in variants], weights=[p for _, p in variants]
            )[0]
            description = bank_label(rng, fill_template(rng, label_pattern, entity, tx_date))

            remarks = ""
            if remarks_pattern is not None and rng.random() < REMARKS_RATE:
                remarks = format_comment_html(fill_template(rng, remarks_pattern, entity, tx_date))
            reference = f"REF{rng.randint(100000, 999999)}" if rng.random() < 0.15 else ""

            lines = (Line(recorded_account(rng, primary), cents, is_debit),)
            transactions.append(
                Transaction(primary, cents, tx_date, description, remarks, reference, lines)
            )
    rng.shuffle(transactions)
    return transactions


FIELDNAMES = [
    "account_code",
    "description",
    "reference",
    "remarks",
    "credit",
    "debit",
    "posting_date",
]


def write_csv(transactions: list[Transaction], path: Path) -> None:
    """Write one row per journal line; a transaction's lines share every text field and the date."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for transaction in transactions:
            for line in transaction.lines:
                amount = f"{line.cents // 100}.{line.cents % 100:02d}"
                writer.writerow(
                    {
                        "account_code": line.account_code,
                        "description": transaction.description,
                        "reference": transaction.reference,
                        "remarks": transaction.remarks,
                        "credit": "0" if line.is_debit else amount,
                        "debit": amount if line.is_debit else "0",
                        "posting_date": transaction.posting_date.isoformat(),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic bank journal.")
    parser.add_argument("--classes", type=int, default=DEFAULT_CLASSES)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    try:
        transactions = generate(args.classes, args.rows, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    write_csv(transactions, args.output)
    n_lines = sum(len(t.lines) for t in transactions)
    print(
        f"Generated {n_lines} rows from {len(transactions)} transactions "
        f"across {args.classes} account codes"
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
