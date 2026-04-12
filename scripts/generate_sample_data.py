"""Generate synthetic French accounting transaction data.

Produces a deterministic (seeded) dataset at data/sample.csv that is realistic
enough to exercise all domain features during training, while containing zero
proprietary data.

Usage:
    python scripts/generate_sample_data.py
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample.csv"
N_ROWS = 7500
DATE_START = date(2024, 1, 1)
DATE_END = date(2025, 12, 31)

# ---------------------------------------------------------------------------
# French PCG accounting codes with target counts (power-law distribution)
# ---------------------------------------------------------------------------
ACCOUNT_CODES: dict[str, int] = {
    # High frequency (370+ samples each)
    "401000": 556,
    "411000": 494,
    "601100": 432,
    "641000": 395,
    "512000": 370,
    # Medium frequency (99-222 each)
    "401100": 222,
    "411100": 210,
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
    "401200": 68,
    "403000": 62,
    "404000": 56,
    "408000": 49,
    "411200": 68,
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
TEMPLATES: dict[str, list[tuple[str, str | None, bool, tuple[float, float]]]] = {
    # --- 401xxx: Supplier payments (debit) ---
    "401000": [
        (
            "PRLV SEPA {entity}",
            "PRLV SEPA CPY:{ref8} RUM:{ref8} NBE:{entity} LIB:REGLEMENT FACTURE",
            True,
            (100, 15000),
        ),
        (
            "VIR SEPA {entity}",
            "VIR SEPA REF:{ref8} NPY:{entity} LIB:COMMANDE FOURNITURES",
            True,
            (200, 25000),
        ),
        (
            "PRLV SEPA {entity}",
            "PRLV SEPA CPY:{ref8} NBE:{entity} LIB:FACTURE {inv}",
            True,
            (50, 8000),
        ),
    ],
    "401100": [
        (
            "VIR SEPA {entity}",
            "VIR SEPA REF:{ref8} NPY:{entity} LIB:FACTURE FOURNISSEUR {inv}",
            True,
            (150, 12000),
        ),
        ("PRLV SEPA {entity}", "PRLV SEPA NBE:{entity} LIB:REGLEMENT ACHAT", True, (80, 5000)),
    ],
    "401200": [
        (
            "VIR SEPA {entity}",
            "VIR SEPA REF:{ref8} NPY:{entity} LIB:ACOMPTE FOURNISSEUR",
            True,
            (500, 20000),
        ),
    ],
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
    # --- 411xxx: Client receipts (credit) ---
    "411000": [
        (
            "VIR RECU {entity}",
            "VIR SEPA NPY:{entity} LIB:REGLEMENT FACTURE {inv}",
            False,
            (500, 50000),
        ),
        ("ENCAISSEMENT CB", "REMISE CB DU {date6} PAYPAL", False, (10, 2000)),
        ("REMISE CHEQUE", "REMISE CHQ N {ref6}", False, (100, 15000)),
    ],
    "411100": [
        (
            "VIR RECU {entity}",
            "VIR SEPA NPY:{entity} LIB:REGLEMENT CLIENT {inv}",
            False,
            (200, 30000),
        ),
    ],
    "411200": [
        ("ENCAISSEMENT CB", "REMISE CB DU {date6} LIB:ENCAISSEMENT CARTE", False, (15, 3000)),
    ],
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def random_date() -> date:
    """Generate a random date between DATE_START and DATE_END.

    40% of dates are clustered near month-end (days 25-31) to mimic
    real treasury patterns (salary, social charges, rent).
    """
    delta_days = (DATE_END - DATE_START).days
    if random.random() < 0.4:
        # Month-end clustering
        year = random.choice([2024, 2025])
        month = random.randint(1, 12)
        day = random.randint(25, 31)
        # Clamp to valid day
        try:
            return date(year, month, day)
        except ValueError:
            # Day out of range for month -- use last day
            next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
            return next_month - timedelta(days=1)
    else:
        offset = random.randint(0, delta_days)
        return DATE_START + timedelta(days=offset)


def fill_template(template: str, entity: str, tx_date: date) -> str:
    """Replace placeholders in a template string."""
    result = template
    result = result.replace("{entity}", entity)
    result = result.replace("{ref8}", f"FR{random.randint(10000000, 99999999)}")
    result = result.replace("{ref6}", str(random.randint(100000, 999999)))
    result = result.replace("{inv}", f"FAC{random.randint(2024000, 2025999)}")
    result = result.replace("{date6}", tx_date.strftime("%d%m%y"))
    result = result.replace("{month}", MONTHS_FR[tx_date.month - 1])
    result = result.replace("{quarter}", QUARTERS_FR[(tx_date.month - 1) // 3])
    result = result.replace("{year}", str(tx_date.year))
    return result


def generate_amount(low: float, high: float) -> float:
    """Generate a transaction amount with 20% round-amount bias."""
    if random.random() < 0.2:
        # Round amount
        magnitude = random.choice([10, 50, 100, 500, 1000])
        amount = magnitude * random.randint(1, max(1, int(high / magnitude)))
        return float(max(low, min(high, amount)))
    else:
        return round(random.uniform(low, high), 2)


def format_comment_html(comment: str) -> str:
    """Format comment with HTML <br /> tags, splitting on LIB: markers."""
    if " LIB:" in comment:
        parts = comment.split(" LIB:", 1)
        return f"{parts[0]}<br />LIB:{parts[1]}"
    return comment


def pick_entity(account_code: str) -> str:
    """Pick a contextually appropriate entity name for an account code."""
    prefix = account_code[:3]
    if prefix in (
        "411",
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
        return random.choice(CLIENT_ENTITIES)
    elif prefix in ("627",):
        return random.choice(SERVICE_ENTITIES[:8])  # Telecom/utilities
    elif prefix in ("616",):
        return random.choice(SERVICE_ENTITIES[8:12])  # Insurance
    elif prefix in ("601", "606", "604", "618", "623", "624", "625", "628"):
        return random.choice(SUPPLIER_ENTITIES + SERVICE_ENTITIES)
    else:
        return random.choice(SUPPLIER_ENTITIES)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def main() -> None:
    rows: list[dict[str, str]] = []

    for account_code, count in ACCOUNT_CODES.items():
        templates = TEMPLATES.get(account_code)
        if templates is None:
            # Fallback: generic template for codes without specific templates
            templates = [
                (
                    "VIR SEPA {entity}",
                    "VIR SEPA NPY:{entity} LIB:OPERATION DIVERSE",
                    True,
                    (100, 10000),
                ),
            ]

        for _ in range(count):
            tpl = random.choice(templates)
            description_pattern, remarks_pattern, is_debit, (amt_low, amt_high) = tpl

            entity = pick_entity(account_code)
            tx_date = random_date()
            amount = generate_amount(amt_low, amt_high)

            description = fill_template(description_pattern, entity, tx_date)

            # ~30% of rows have remarks (matching real-world distribution)
            if remarks_pattern is not None and random.random() < 0.30:
                remarks_raw = fill_template(remarks_pattern, entity, tx_date)
                remarks = format_comment_html(remarks_raw)
            else:
                remarks = ""

            # ~15% of rows get a reference number
            reference = ""
            if random.random() < 0.15:
                reference = f"REF{random.randint(100000, 999999)}"

            if is_debit:
                credit, debit = 0, amount
            else:
                credit, debit = amount, 0

            rows.append(
                {
                    "account_code": account_code,
                    "description": description,
                    "reference": reference,
                    "remarks": remarks,
                    "credit": f"{credit:.2f}" if credit else "0",
                    "debit": f"{debit:.2f}" if debit else "0",
                    "posting_date": tx_date.isoformat(),
                }
            )

    # Shuffle deterministically (seed already set)
    random.shuffle(rows)

    # Write CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "account_code",
        "description",
        "reference",
        "remarks",
        "credit",
        "debit",
        "posting_date",
    ]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = sum(ACCOUNT_CODES.values())
    print(f"Generated {len(rows)} rows ({total} target) across {len(ACCOUNT_CODES)} account codes")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
