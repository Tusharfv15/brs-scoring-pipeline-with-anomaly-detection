RECENCY_MAP = {
    # Days approximation
    "Just now": 0, "a minute ago": 0, "2 minutes ago": 0,
    "3 minutes ago": 0, "4 minutes ago": 0, "5 minutes ago": 0,
    "10 minutes ago": 0, "15 minutes ago": 0, "20 minutes ago": 0,
    "30 minutes ago": 0, "45 minutes ago": 0,
    "an hour ago": 0, "2 hours ago": 0, "3 hours ago": 0,
    "4 hours ago": 0, "5 hours ago": 0, "6 hours ago": 0,
    "7 hours ago": 0, "8 hours ago": 0, "9 hours ago": 0,
    "10 hours ago": 0, "11 hours ago": 0, "12 hours ago": 0,
    "13 hours ago": 0, "14 hours ago": 0, "15 hours ago": 0,
    "16 hours ago": 0, "17 hours ago": 0, "18 hours ago": 0,
    "19 hours ago": 0, "20 hours ago": 0, "21 hours ago": 0,
    "22 hours ago": 0, "23 hours ago": 0,
    "yesterday": 1,
    "2 days ago": 2, "3 days ago": 3, "4 days ago": 4,
    "5 days ago": 5, "6 days ago": 6,
    "a week ago": 7, "2 weeks ago": 14,
    "3 weeks ago": 21, "4 weeks ago": 28,
    "a month ago": 30, "2 months ago": 60, "3 months ago": 90,
    "4 months ago": 120, "5 months ago": 150, "6 months ago": 180,
    "7 months ago": 210, "8 months ago": 240, "9 months ago": 270,
    "10 months ago": 300, "11 months ago": 330,
    "a year ago": 365, "2 years ago": 730, "3 years ago": 1095,
    "4 years ago": 1460, "5 years ago": 1825, "6 years ago": 2190,
    "7 years ago": 2555, "8 years ago": 2920, "9 years ago": 3285,
    "10 years ago": 3650,
}

def recency_multiplier(days: int) -> float:
    if days <= 90:    return 1.00   # < 3 months
    if days <= 365:   return 0.85   # 3–12 months
    if days <= 730:   return 0.65   # 1–2 years
    if days <= 1095:  return 0.45   # 2–3 years
    return 0.25                      # 3+ years