"""All hand-curated mapping tables live here so reviewers can audit them in one place."""

# ---------------------------------------------------------------- sectors
# 48 raw labels -> 14 groups (+ "Unknown")
SECTOR_MAP = {
    "Consumer & Retail": ["Consumer", "Retail", "Consumer Goods"],
    "Fintech": ["Fintech", "Financial Services", "Unbundling of Bank",
                "Trending Themes In Payments", "Investment Industry",
                "Trending Themes In Investment Tech", "Blockchain Technology"],
    "Food & Agri": ["Food and Agriculture Tech", "Food and Agriculture",
                    "Trending Themes In Food Tech"],
    "Enterprise & SaaS": ["Enterprise Applications", "Business Services",
                          "Enterprise Infrastructure"],
    "Deep Tech & Hardware": ["High Tech", "Semiconductors", "AI-native", "Mobile",
                             "Telecom", "Internet of Things - Industry Applications"],
    "Healthtech": ["Healthtech", "Healthcare", "Life Sciences"],
    "Edtech": ["Edtech", "Education"],
    "Energy & Climate": ["Energy Tech", "Energy", "Environment Tech", "Environment",
                         "Sustainability Tech"],
    "Real Estate & Construction": ["Real Estate and Construction Tech",
                                   "Real Estate and Construction"],
    "Media & Gaming": ["Media & Entertainment", "Gaming"],
    "Mobility & Logistics": ["Auto", "Auto Tech", "Transportation and Logistics"],
    "Industrial & Manufacturing": ["Industrial Goods and Manufacturing",
                                   "Chemicals and Materials", "Chemicals and Materials Tech",
                                   "Aerospace, Maritime and Defense Tech",
                                   "Aerospace, Maritime and Defense"],
    "Travel & Hospitality": ["Travel and Hospitality", "Travel and Hospitality Tech"],
    "Other": ["Nonprofits", "Gig Economy"],
}
SECTOR_LOOKUP = {raw: grp for grp, raws in SECTOR_MAP.items() for raw in raws}

# ---------------------------------------------------------------- stages
STAGE_MAP = {
    "Pre-seed": "Seed", "Seed": "Seed", "Angel": "Seed", "Grant": "Seed",
    "Convertible note": "Seed", "Crowdfunding": "Seed",
    "Pre-Series A": "Series A", "Series A": "Series A",
    "Pre-Series B": "Series B", "Series B": "Series B",
    "Pre-Series C": "Growth", "Series C": "Growth", "Pre-Series D": "Growth",
    "Series D": "Growth", "Series E": "Growth", "Series F": "Growth",
    "Series G": "Growth", "Series H": "Growth", "Growth / PE": "Growth",
    "Debt": "Debt", "Venture debt": "Debt",
    "Bridge": "Other", "Strategic": "Other",
    "Undisclosed": "Undisclosed",
}
STAGE_ORDER = ["Seed", "Series A", "Series B", "Growth", "Debt", "Other", "Undisclosed"]

# ---------------------------------------------------------------- cities
CITY_ALIASES = {
    "bangalore": "Bengaluru", "bangluru": "Bengaluru", "bengluru": "Bengaluru",
    "bangalore rural": "Bengaluru", "bangalore/bengaluru": "Bengaluru",
    "gurgaon": "Gurugram", "guargaon": "Gurugram", "guarguron": "Gurugram",
    "guragram": "Gurugram", "gurggram": "Gurugram", "guruguram": "Gurugram",
    "gurgaon and bangalore": "Gurugram",
    "new-delhi": "New Delhi", "delhi": "New Delhi", "delhi-ncr": "New Delhi",
    "delhi-ncr region": "New Delhi", "delhi/ncr": "New Delhi",
    "new delhi and new york": "New Delhi", "dwarka": "New Delhi",
    "mysore": "Mysuru", "mangalore": "Mangaluru", "bhubaneshwar": "Bhubaneswar",
    "secunderabad": "Hyderabad", "gachibowli": "Hyderabad",
    "sahibzada ajit singh nagar": "Mohali", "gift city": "Gandhinagar",
    "ahmedabad/bengaluru": "Ahmedabad", "bengaluru/mumbai": "Bengaluru",
    "dona paula": "Goa", "panaji": "Goa", "bardez": "Goa", "mormugao": "Goa",
    "trichy": "Tiruchirappalli",
}
NCR = {"New Delhi", "Gurugram", "Noida", "Greater Noida", "Ghaziabad", "Faridabad"}
HUBS = {"Bengaluru", "Mumbai", "Hyderabad", "Chennai", "Pune", "Ahmedabad", "Kolkata"}
STATES_NOT_CITIES = {"Gujarat", "Kerala", "Maharashtra", "Jharkhand", "Rajasthan",
                     "Uttar Pradesh"}
INTERNATIONAL = {"Singapore", "Dubai", "New York", "San Francisco", "Barcelona",
                 "Bedford", "Columbia", "Sunnyvale"}

# ---------------------------------------------------------------- investors
# Free-text entries that are NOT a single identifiable investor.
JUNK_INVESTOR_PATTERNS = [
    r"^(few|several|various|multiple|other|others|unknown|undisclosed|n/?a|none|nil)\b",
    r"\b(overseas investors|private investors|existing investors|new investors|"
    r"high net worth|hnis?|family offices?|consortium|angel investors|"
    r"partners and employees|marquee (firms|investors)|a group of|group of)\b",
    r"\b(managing director|md of|ceo|founder of|co-founder|chairman)\b",
    r"^(individual|individuals)$",
    r"^(us|uk|foreign|international|overseas|singapore|gulf)[- ]based (investor|investors|fund|funds)$",
]
# Same firm under different names (rebrands etc.)  key = normalised alias -> canonical
MANUAL_INVESTOR_ALIASES = {
    "peak xv partners": "Sequoia Capital India",   # Sequoia India rebranded to Peak XV in 2023
    "peak xv": "Sequoia Capital India",
    "sequoia capital": "Sequoia Capital India",
    "sequoia": "Sequoia Capital India",
    "tiger global": "Tiger Global Management",
    # confirmed by manual review of reports/investor_near_duplicates_for_review.csv
    "neev ii fund": "Neev Fund II",
    "aavishkar capital": "Aavishkaar Capital",
    "venture catalyst": "Venture Catalysts",
    "iifl s fintech fund": "IIFL Fintech Fund",
    "softbank vision fund 2": "SoftBank",
    "softbank vision fund": "SoftBank",
    "softbank group": "SoftBank",
    "japan s softbank": "SoftBank",
    "softbank": "SoftBank",
    "matrix partner india": "Matrix Partners India",
    "z47 formerly matrix partners india": "Matrix Partners India",
    "matrix": "Matrix Partners India",
    "inflection point ip ventures": "Inflection Point Ventures",
    "matrix partners india devc": "Matrix Partners India",
}
# tokens ignored when building the matching key (legal suffixes / generic words)
KEY_STOPWORDS = {"india", "management", "pvt", "ltd", "llp", "limited", "private",
                 "inc", "corp", "corporation", "the", "advisors", "advisers", "company"}

INSTITUTION_KEYWORDS = ["foundation", "authority", "department", "ministry", "sidbi",
                        "nabard", "government", "council", "bank", "dst", "nsdc",
                        "national skill", "iit", "iim", "incubat", "accelerator",
                        "corporation", "university"]

# ---------------------------------------------------------------- numeric bins
AMOUNT_BINS = [0, 1e5, 1e6, 5e6, 2e7, 1e8, float("inf")]
AMOUNT_LABELS = ["<$100K", "$100K-1M", "$1M-5M", "$5M-20M", "$20M-100M", "$100M+"]
AGE_BINS = [-1, 2, 5, 9, 200]
AGE_LABELS = ["0-2y", "3-5y", "6-9y", "10y+"]

MIN_DEALS = 3          # investors with fewer deals are not modelled
RANDOM_SEED = 42
