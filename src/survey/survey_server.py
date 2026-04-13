#!/usr/bin/env python3
"""
Simple A/B image quality survey for anomaly generation methods.
Run with:  uv run src/survey_html/survey_server.py
"""

import csv
import io
import itertools
import os
import random
from datetime import datetime
from pathlib import Path

from PIL import Image

from flask import (
    Flask,
    abort,
    redirect,
    render_template_string,
    request,
    send_file,
    session,
    url_for,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SURVEY_SECRET", "change-me-in-production-xyz-123")

# In-memory last-seen tracker: {email: datetime}
_last_seen: dict[str, datetime] = {}

GENERATED_DATASETS_DIR = Path("generated_datasets").resolve()
DATASETS_DIR = Path("datasets").resolve()
SURVEY_DIR = Path(__file__).parent
OUTPUTS_DIR = SURVEY_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
CSV_PATH = OUTPUTS_DIR / "survey_results.csv"

METHODS = ["glass", "olga_anomalyany", "ourmethod", "realnet", "real_NOT_GEN"]

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
DATASET_CATEGORIES = {"mvtec": MVTEC_CATEGORIES, "visa": VISA_CATEGORIES}

MAX_QUERIES = 50

CSV_FIELDS = [
    "email", "timestamp", "query_num", "dataset", "category",
    "method_a", "method_b", "img_a_path", "img_b_path", "vote",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare_image(path: Path) -> io.BytesIO:
    """Resize to 512x512 and encode as JPEG q=80."""
    img = Image.open(path).convert("RGB")
    img = img.resize((512, 512), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    return buf


def get_images_for(method: str, dataset: str, category: str) -> list[Path]:
    img_dir = GENERATED_DATASETS_DIR / method / dataset / category / "images"
    if not img_dir.exists():
        return []
    return [p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]


def get_available_methods(dataset: str, category: str) -> list[str]:
    return [m for m in METHODS if get_images_for(m, dataset, category)]


def count_user_votes(email: str) -> int:
    """Return how many votes this email has already recorded in the CSV."""
    if not CSV_PATH.exists():
        return 0
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for row in reader if row["email"] == email)


def read_pair_counts() -> dict:
    """Return {(dataset, category, frozenset{m1, m2}): count} from the CSV."""
    counts: dict = {}
    if not CSV_PATH.exists():
        return counts
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["category"], frozenset({row["method_a"], row["method_b"]}))
            counts[key] = counts.get(key, 0) + 1
    return counts


def sample_question() -> dict | None:
    """Sample a question with 50/50 dataset stratification and inverse-frequency weighting.

    This re-checks the filesystem on every call, so new images added without
    restarting the server are picked up automatically.
    Dataset is chosen 50/50 (mvtec vs visa) first; then inverse-frequency
    weighting (1 / (count + 1)) is applied within that dataset's combos.
    """
    counts = read_pair_counts()

    # 50/50 dataset stratification, then inverse-frequency within dataset
    datasets = ["mvtec", "visa"]
    random.shuffle(datasets)          # try chosen dataset first, fallback to other
    for ds in datasets:
        combos = []
        for cat in DATASET_CATEGORIES[ds]:
            available = get_available_methods(ds, cat)
            if len(available) < 2:
                continue
            for m1, m2 in itertools.combinations(available, 2):
                combos.append((ds, cat, m1, m2))
        if not combos:
            continue

        weights = [
            1.0 / (counts.get((ds, cat, frozenset({m1, m2})), 0) + 1)
            for ds, cat, m1, m2 in combos
        ]

        for _ in range(10):                    # guard against transient fs issues
            dataset, category, method_a, method_b = random.choices(combos, weights=weights, k=1)[0]
            if random.random() < 0.5:
                method_a, method_b = method_b, method_a
            imgs_a = get_images_for(method_a, dataset, category)
            imgs_b = get_images_for(method_b, dataset, category)
            if not imgs_a or not imgs_b:
                continue
            img_a = random.choice(imgs_a)
            img_b = random.choice(imgs_b)
            return {
                "dataset": dataset, "category": category,
                "method_a": method_a, "method_b": method_b,
                "img_a": str(img_a.relative_to(GENERATED_DATASETS_DIR)),
                "img_b": str(img_b.relative_to(GENERATED_DATASETS_DIR)),
            }
    return None


def log_vote(email: str, query_num: int, question: dict, vote: str) -> None:
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "email": email,
            "timestamp": datetime.utcnow().isoformat(),
            "query_num": query_num,
            "dataset": question["dataset"],
            "category": question["category"],
            "method_a": question["method_a"],
            "method_b": question["method_b"],
            "img_a_path": str(GENERATED_DATASETS_DIR / question["img_a"]),
            "img_b_path": str(GENERATED_DATASETS_DIR / question["img_b"]),
            "vote": vote,
        })


def compute_debug_stats() -> dict | None:
    if not CSV_PATH.exists():
        return None
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    from collections import Counter, defaultdict

    total = len(rows)
    ties  = sum(1 for r in rows if r["vote"] == "TIE")

    # --- category distribution ---
    cat_counter = Counter((r["dataset"], r["category"]) for r in rows)
    cat_items   = sorted(cat_counter.items(), key=lambda x: x[1], reverse=True)
    cat_labels  = [f"{ds}/{cat}" for (ds, cat), _ in cat_items]
    cat_data    = [cnt for _, cnt in cat_items]
    cat_colors  = [
        "rgba(99,102,241,0.75)" if ds == "mvtec" else "rgba(234,179,8,0.75)"
        for (ds, _), _ in cat_items
    ]

    # --- method appearances & wins ---
    method_appearances: Counter = Counter()
    method_wins:        Counter = Counter()
    for r in rows:
        method_appearances[r["method_a"]] += 1
        method_appearances[r["method_b"]] += 1
        if r["vote"] == "A":
            method_wins[r["method_a"]] += 1
        elif r["vote"] == "B":
            method_wins[r["method_b"]] += 1

    method_labels      = sorted(method_appearances, key=lambda m: -method_appearances[m])
    method_appear_data = [method_appearances[m] for m in method_labels]
    method_win_pct     = [
        round(method_wins.get(m, 0) / method_appearances[m] * 100, 1)
        for m in method_labels
    ]

    # --- per-method category breakdown (for stacked chart) ---
    # method_cat[method][cat_label] = count of appearances
    method_cat: dict = defaultdict(Counter)
    for r in rows:
        label = f"{r['dataset']}/{r['category']}"
        method_cat[r["method_a"]][label] += 1
        method_cat[r["method_b"]][label] += 1

    method_cat_datasets = []
    palette = ["#6366f1", "#16a34a", "#dc2626", "#ca8a04", "#0891b2"]
    for i, m in enumerate(method_labels):
        method_cat_datasets.append({
            "label": m,
            "data": [method_cat[m].get(lbl, 0) for lbl in cat_labels],
            "backgroundColor": palette[i % len(palette)],
            "borderRadius": 2,
        })

    # --- pair breakdown table ---
    pair_data: dict = defaultdict(lambda: {"n": 0, "wins": defaultdict(int), "ties": 0})
    for r in rows:
        key = tuple(sorted([r["method_a"], r["method_b"]]))
        pair_data[key]["n"] += 1
        if r["vote"] == "A":
            pair_data[key]["wins"][r["method_a"]] += 1
        elif r["vote"] == "B":
            pair_data[key]["wins"][r["method_b"]] += 1
        else:
            pair_data[key]["ties"] += 1

    pair_rows = [
        {
            "m1": m1, "m2": m2,
            "n":  d["n"],
            "m1_wins": d["wins"].get(m1, 0),
            "m2_wins": d["wins"].get(m2, 0),
            "ties":    d["ties"],
        }
        for (m1, m2), d in sorted(pair_data.items(), key=lambda x: -x[1]["n"])
    ]

    # --- user table ---
    user_rows = Counter(r["email"] for r in rows).most_common()

    return dict(
        total=total, ties=ties,
        tie_pct=round(ties / total * 100, 1),
        n_users=len(user_rows), n_cats=len(cat_counter),
        cat_labels=cat_labels, cat_data=cat_data, cat_colors=cat_colors,
        method_labels=method_labels,
        method_appear_data=method_appear_data, method_win_pct=method_win_pct,
        method_cat_datasets=method_cat_datasets,
        pair_rows=pair_rows, user_rows=user_rows,
    )


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_BASE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Anomaly Image Survey</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0d0d0d;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
  }
  .container { width: 100%; max-width: 960px; }
  h1 { font-size: 1.65rem; font-weight: 700; margin-bottom: 0.4rem; }
  p { color: #888; line-height: 1.65; margin-bottom: 1.2rem; }
  strong { color: #e0e0e0; }
  input[type=email] {
    width: 100%;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    border: 1px solid #2e2e2e;
    background: #1a1a1a;
    color: #e0e0e0;
    font-size: 1rem;
    margin-bottom: 1rem;
    outline: none;
    transition: border-color 0.2s;
  }
  input[type=email]:focus { border-color: #6366f1; }
  .btn {
    display: inline-block;
    padding: 0.75rem 2rem;
    border-radius: 8px;
    border: none;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: filter 0.15s;
    text-decoration: none;
  }
  .btn:hover { filter: brightness(1.15); }
  .btn-primary { background: #6366f1; color: #fff; width: 100%; }
  .btn-a {
    background: #2563eb; color: #fff;
    width: 100%; margin-top: 0.6rem;
    font-size: 1.05rem; padding: 0.85rem;
  }
  .btn-b {
    background: #16a34a; color: #fff;
    width: 100%; margin-top: 0.6rem;
    font-size: 1.05rem; padding: 0.85rem;
  }
  .btn-skip {
    background: #991b1b; color: #fff;
    width: 100%; margin-top: 0.6rem;
    font-size: 0.95rem; padding: 0.75rem;
  }
  .image-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
    margin: 1.25rem 0;
  }
  .image-card {
    background: #141414;
    border: 1px solid #242424;
    border-radius: 12px;
    overflow: hidden;
  }
  .image-card img {
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: contain;
    background: #0a0a0a;
    display: block;
  }
  .image-label {
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: #555;
    text-align: center;
    letter-spacing: 0.05em;
  }
  .progress-wrap { margin-bottom: 1.25rem; }
  .progress-meta { font-size: 0.78rem; color: #555; margin-bottom: 0.35rem; }
  .progress-bar-bg {
    height: 4px; background: #1f1f1f; border-radius: 2px;
  }
  .progress-bar-fill {
    height: 100%; background: #6366f1; border-radius: 2px; transition: width 0.3s;
  }
  .vote-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .center { text-align: center; }
  .ref-link { color: #6366f1; font-size: 0.82rem; text-decoration: none; opacity: 0.8; }
  .ref-link:hover { opacity: 1; text-decoration: underline; }
  .big-icon { font-size: 3.5rem; margin-bottom: 1rem; }
  @media (max-width: 600px) {
    .image-grid, .vote-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="container">
{% block body %}{% endblock %}
</div>
</body>
</html>"""

INDEX_TMPL = _BASE.replace("{% block body %}{% endblock %}", """
<div style="max-width:460px;margin:0 auto">
  <h1>Anomaly Image Quality Survey</h1>
  <p style="margin-top:0.6rem">
    You will see <strong>50 pairs</strong> of synthetic anomaly images generated
    by different methods. For each pair, pick the image whose anomalies look
    <strong>more realistic and naturally defective</strong>.
  </p>
  <p>Takes roughly 5&ndash;8 minutes. Thank you!</p>
  <form method="POST" action="/start">
    <input type="email" name="email" placeholder="your@email.com" required autofocus />
    <button type="submit" class="btn btn-primary">Start &rarr;</button>
  </form>
</div>
""")

SURVEY_TMPL = _BASE.replace("{% block body %}{% endblock %}", """
<div class="progress-wrap">
  <div class="progress-meta">Question {{ query_num }} / {{ max_queries }} &nbsp;&middot;&nbsp; <span style="text-transform:uppercase">{{ dataset }}</span>: {{ category }}</div>
  <div class="progress-bar-bg">
    <div class="progress-bar-fill" style="width:{{ pct }}%"></div>
  </div>
</div>
<h1>Which image shows more realistic anomalies?</h1>
<p>Pick the image that shows the most plausible and realistic defect.
  <br/><a href="#" class="ref-link" onclick="showRef(event)">&#128247; See a defect-free reference image</a>
</p>
<div id="ref-modal" onclick="this.style.display='none'" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.82);z-index:999;align-items:center;justify-content:center;cursor:zoom-out">
  <img id="ref-img" src="" alt="Reference" style="max-width:90vw;max-height:85vh;border-radius:10px;box-shadow:0 0 40px #000" onclick="event.stopPropagation()" />
</div>
<script>
function showRef(e) {
  e.preventDefault();
  var m = document.getElementById('ref-modal');
  document.getElementById('ref-img').src = '/ref/{{ dataset }}/{{ category }}?' + Date.now();
  m.style.display = 'flex';
}
</script>
<div class="image-grid">
  <div class="image-card">
    <img src="/img/{{ img_a }}" alt="Image A" loading="eager" />
    <div class="image-label">A</div>
  </div>
  <div class="image-card">
    <img src="/img/{{ img_b }}" alt="Image B" loading="eager" />
    <div class="image-label">B</div>
  </div>
</div>
<div class="vote-grid">
  <form method="POST" action="/vote">
    <input type="hidden" name="vote" value="A" />
    <button type="submit" class="btn btn-a">Defect in "A" is more realistic & plausible</button>
  </form>
  <form method="POST" action="/vote">
    <input type="hidden" name="vote" value="B" />
    <button type="submit" class="btn btn-b">Defect in "B" is more realistic & plausible</button>
  </form>
</div>
<div style="margin-top:0.5rem">
  <form method="POST" action="/vote">
    <input type="hidden" name="vote" value="TIE" />
    <button type="submit" class="btn btn-skip">I really can't decide</button>
  </form>
</div>
""")

END_TMPL = _BASE.replace("{% block body %}{% endblock %}", """
<div class="center" style="max-width:460px;margin:0 auto">
  <div class="big-icon">&#127881;</div>
  <h1>Thank you!</h1>
  <p style="margin-top:0.75rem;font-size:1.05rem">
    You completed all {{ total_done }} comparisons.<br/>
    Your responses have been recorded &mdash; we really appreciate your time!
  </p>
</div>
""")


DEBUG_TMPL = _BASE.replace("{% block body %}{% endblock %}", """
<style>
  .stats-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:2rem; }
  .stat-card  { background:#1a1a1a; border:1px solid #242424; border-radius:10px; padding:1.25rem; }
  .stat-label { font-size:0.72rem; color:#555; text-transform:uppercase; letter-spacing:.06em; margin-bottom:.35rem; }
  .stat-value { font-size:2rem; font-weight:700; }
  h2 { font-size:.85rem; color:#555; text-transform:uppercase; letter-spacing:.07em; margin-bottom:1rem; }
  .card { background:#1a1a1a; border:1px solid #242424; border-radius:10px; padding:1.5rem; margin-bottom:2rem; }
  table { width:100%; border-collapse:collapse; font-size:.82rem; }
  th { text-align:left; color:#555; font-weight:600; padding:.45rem .75rem; border-bottom:1px solid #222; }
  td { padding:.4rem .75rem; border-bottom:1px solid #191919; }
  tr:hover td { background:#161616; }
  .pct { color:#555; }
  @media(max-width:700px){ .stats-grid{ grid-template-columns:1fr 1fr; } }
</style>
<div style="max-width:1100px;margin:0 auto">
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.75rem">
  <h1>Survey Dashboard</h1>
  <div style="display:flex;align-items:center;gap:1rem">
    <span style="font-size:.8rem;color:#555">auto-refresh</span>
    <a href="/debug" style="color:#555;font-size:.85rem;text-decoration:none">&#8635;</a>
    <a href="/" style="color:#555;font-size:.85rem;text-decoration:none">&#8592; Back</a>
  </div>
</div>

<h2>Live activity</h2>
<div class="card" style="margin-bottom:2rem">
  {% if active_now %}
  <table>
    <thead><tr><th>User</th><th>Last action</th><th>Status</th></tr></thead>
    <tbody>
    {% for email, idle_s in active_now %}
    <tr>
      <td>{{ email }}</td>
      <td>
        {% if idle_s < 60 %}{{ idle_s }}s ago
        {% else %}{{ (idle_s // 60)|int }}m {{ (idle_s % 60)|int }}s ago{% endif %}
      </td>
      <td>
        {% if idle_s < 120 %}
          <span style="color:#16a34a;font-weight:600">&#9679; active</span>
        {% elif idle_s < 600 %}
          <span style="color:#ca8a04">&#9679; idle</span>
        {% else %}
          <span style="color:#444">&#9679; away</span>
        {% endif %}
      </td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
  <p style="color:#444;margin:0">No one has loaded a page since the server started.</p>
  {% endif %}
</div>

<div class="stats-grid">
  <div class="stat-card"><div class="stat-label">Total votes</div><div class="stat-value">{{ total }}</div></div>
  <div class="stat-card"><div class="stat-label">Unique users</div><div class="stat-value">{{ n_users }}</div></div>
  <div class="stat-card"><div class="stat-label">TIE rate</div><div class="stat-value">{{ tie_pct }}%</div></div>
  <div class="stat-card"><div class="stat-label">Categories seen</div><div class="stat-value">{{ n_cats }}/27</div></div>
</div>

<h2>Category sampling : per method (stacked)</h2>
<div class="card">
  <div style="position:relative;height:{{ [cat_labels|length * 22 + 40, 200]|max }}px">
    <canvas id="catChart"></canvas>
  </div>
</div>

<h2>Method appearances &amp; win rate</h2>
<div class="card">
  <div style="position:relative;height:180px"><canvas id="methodChart"></canvas></div>
</div>

<h2>Method pair breakdown</h2>
<div class="card">
  <table>
    <thead><tr><th>Pair</th><th>Comparisons</th><th colspan="2">Wins</th><th>Ties</th></tr></thead>
    <tbody>
    {% for r in pair_rows %}
    <tr>
      <td>{{ r.m1 }} <span class="pct">vs</span> {{ r.m2 }}</td>
      <td>{{ r.n }}</td>
      <td>{{ r.m1 }}: {{ r.m1_wins }} <span class="pct">({{ (r.m1_wins / r.n * 100)|round(1) }}%)</span></td>
      <td>{{ r.m2 }}: {{ r.m2_wins }} <span class="pct">({{ (r.m2_wins / r.n * 100)|round(1) }}%)</span></td>
      <td>{{ r.ties }}</td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

<h2>Votes per user</h2>
<div class="card">
  <table>
    <thead><tr><th>Email</th><th>Votes</th></tr></thead>
    <tbody>
    {% for email, cnt in user_rows %}
    <tr><td>{{ email }}</td><td>{{ cnt }}</td></tr>
    {% endfor %}
    </tbody>
  </table>
</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script>
Chart.defaults.color = '#666';
Chart.defaults.borderColor = '#222';

new Chart(document.getElementById('catChart'), {
  type: 'bar',
  data: {
    labels: {{ cat_labels | tojson }},
    datasets: {{ method_cat_datasets | tojson }},
  },
  options: {
    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color:'#888', boxWidth:12 } } },
    scales: {
      x: { stacked:true, grid:{color:'#1f1f1f'}, ticks:{color:'#555'} },
      y: { stacked:true, grid:{display:false}, ticks:{color:'#888', font:{size:11}} }
    }
  }
});

new Chart(document.getElementById('methodChart'), {
  type: 'bar',
  data: {
    labels: {{ method_labels | tojson }},
    datasets: [
      { label:'Appearances', data:{{ method_appear_data | tojson }},
        backgroundColor:'rgba(99,102,241,0.7)', borderRadius:4, yAxisID:'y' },
      { label:'Win %', data:{{ method_win_pct | tojson }},
        backgroundColor:'rgba(22,163,74,0.7)', borderRadius:4, yAxisID:'y2' }
    ]
  },
  options: {
    responsive:true, maintainAspectRatio:false,
    plugins:{ legend:{ labels:{ color:'#888', boxWidth:12 } } },
    scales:{
      y:  { grid:{color:'#1f1f1f'}, ticks:{color:'#555'}, title:{display:true,text:'Appearances',color:'#555'} },
      y2: { position:'right', grid:{display:false}, max:100,
            ticks:{color:'#555', callback: v => v+'%'},
            title:{display:true,text:'Win %',color:'#555'} },
      x:  { grid:{display:false}, ticks:{color:'#888'} }
    }
  }
});
</script>
""")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(INDEX_TMPL)


@app.route("/start", methods=["POST"])
def start():
    email = request.form.get("email", "").strip()
    if not email:
        return redirect(url_for("index"))
    if email == "vad@father.com":
        return redirect(url_for("debug"))
    done = count_user_votes(email)
    target = (done // 50 + 1) * 50
    session.clear()
    session["email"] = email
    session["query_count"] = done
    session["max_queries"] = target
    return redirect(url_for("survey"))


@app.route("/debug")
def debug():
    now = datetime.utcnow()
    active_now = sorted(
        [
            (email, int((now - ts).total_seconds()))
            for email, ts in _last_seen.items()
            if (now - ts).total_seconds() < 1800
        ],
        key=lambda x: x[1],
    )
    stats = compute_debug_stats() or {}
    stats["active_now"] = active_now
    return render_template_string(DEBUG_TMPL, **stats)


@app.route("/survey")
def survey():
    if "email" not in session:
        return redirect(url_for("index"))
    _last_seen[session["email"]] = datetime.utcnow()
    max_q = session.get("max_queries", MAX_QUERIES)
    if session.get("query_count", 0) >= max_q:
        return redirect(url_for("end"))

    # Generate a new question if there isn't one pending
    if "question" not in session:
        q = sample_question()
        if q is None:
            abort(500, "Could not find a valid image pair. Check dataset availability.")
        session["question"] = q

    q = session["question"]
    query_num = session["query_count"] + 1
    pct = round((query_num - 1) / max_q * 100)

    return render_template_string(
        SURVEY_TMPL,
        query_num=query_num,
        max_queries=max_q,
        dataset=q["dataset"],
        category=q["category"],
        img_a=q["img_a"],
        img_b=q["img_b"],
        pct=pct,
    )


@app.route("/vote", methods=["POST"])
def vote():
    if "email" not in session or "question" not in session:
        return redirect(url_for("index"))
    _last_seen[session["email"]] = datetime.utcnow()

    vote_val = request.form.get("vote", "").upper()
    if vote_val not in ("A", "B", "TIE"):
        return redirect(url_for("survey"))

    q = session.pop("question")
    session["query_count"] = session.get("query_count", 0) + 1

    log_vote(
        email=session["email"],
        query_num=session["query_count"],
        question=q,
        vote=vote_val,
    )

    if session["query_count"] >= session.get("max_queries", MAX_QUERIES):
        return redirect(url_for("end"))
    return redirect(url_for("survey"))


@app.route("/end")
def end():
    total_done = session.get("max_queries", MAX_QUERIES)
    return render_template_string(END_TMPL, total_done=total_done)


@app.route("/img/<path:rel_path>")
def serve_image(rel_path: str):
    """Serve preprocessed images from the generated_datasets directory."""
    full_path = (GENERATED_DATASETS_DIR / rel_path).resolve()
    if not str(full_path).startswith(str(GENERATED_DATASETS_DIR)):
        abort(403)
    if not full_path.exists():
        abort(404)
    return send_file(prepare_image(full_path), mimetype="image/jpeg")


@app.route("/ref/<dataset>/<category>")
def serve_reference(dataset: str, category: str):
    """Serve a random defect-free reference image for the given dataset/category."""
    # Validate inputs against known lists to prevent path traversal
    if dataset not in DATASET_CATEGORIES or category not in DATASET_CATEGORIES[dataset]:
        abort(404)
    if dataset == "mvtec":
        ref_dir = DATASETS_DIR / "mvtec" / category / "train" / "good"
    else:
        ref_dir = DATASETS_DIR / "visa" / category / "Data" / "Images" / "Normal"
    images = list(ref_dir.glob("*.png")) + list(ref_dir.glob("*.PNG")) \
           + list(ref_dir.glob("*.jpg")) + list(ref_dir.glob("*.JPG"))
    if not images:
        abort(404)
    return send_file(prepare_image(random.choice(images)), mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Survey server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Survey server starting on http://{args.host}:{args.port}")
    print(f"Results will be saved to: {CSV_PATH}")
    app.run(host=args.host, port=args.port, debug=args.debug)
