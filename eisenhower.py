# app.py
# Eisenhower Project Planner – Bug-fixed & Production-ready
# ---------------------------------------------------------
# 1. Login/Registrierung (bcrypt + SQLite)
# 2. Projekte & Tasks mit Eisenhower-Quadranten
# 3. LLM-Task-Generierung (OpenAI) + deterministische Scoring
# 4. Robustes Caching, saubere DB-Handles, Input-Validierung
# ---------------------------------------------------------
# Run:  streamlit run app.py
# Env:  OPENAI_API_KEY=...   (optional)

from __future__ import annotations

import os
import json
import re
import sqlite3
import hashlib
import secrets
import bcrypt
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd

# --------------- CONFIG ---------------
st.set_page_config(
    page_title="Eisenhower Planner",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------- ENV ---------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "") or "gpt-4o-mini"

try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

# --------------- STYLING ---------------
st.markdown(
    """
<style>
.qhdr {font-weight:800;font-size:1.05rem;margin:0 0 .25rem 0;}
.qsub {opacity:.8;font-size:.85rem;margin:0 0 .75rem 0;}
.qbox {border-radius:18px;padding:14px 14px 8px 14px;border:1px solid rgba(255,255,255,.08);}
.q_du_wi {background:rgba(255,77,77,.10);}
.q_nd_wi {background:rgba(77,140,255,.10);}
.q_du_nwi {background:rgba(255,190,77,.12);}
.q_nd_nwi {background:rgba(160,160,160,.10);}
.tcard {border-radius:14px;padding:10px 12px;margin:0 0 10px 0;border:1px solid rgba(255,255,255,.08);}
.trow {display:flex;gap:10px;align-items:center;justify-content:space-between;}
.tleft {display:flex;flex-direction:column;gap:4px;}
.ttitle {font-weight:750;}
.tmeta {opacity:.85;font-size:.85rem;}
.badge {display:inline-block;padding:3px 8px;border-radius:999px;font-size:.78rem;border:1px solid rgba(255,255,255,.12);}
.b_imp {background:rgba(77,140,255,.12);}
.b_eff {background:rgba(255,190,77,.14);}
.b_urg {background:rgba(255,77,77,.12);}
.b_stat {background:rgba(120,255,160,.10);}
.small {font-size:.82rem;opacity:.9;}
hr {border:none;border-top:1px solid rgba(255,255,255,.08);margin:.6rem 0;}
</style>
""",
    unsafe_allow_html=True,
)

# --------------- CONSTANTS ---------------
QUADRANTS = {
    "DU_WI":  "Dringend & Wichtig (DO)",
    "ND_WI":  "Nicht dringend & Wichtig (PLAN)",
    "DU_NWI": "Dringend & Nicht wichtig (DELEGIEREN)",
    "ND_NWI": "Nicht dringend & Nicht wichtig (ELIMINIEREN)",
}
STATUS_OPTIONS = ["todo", "doing", "done", "blocked"]
DEFAULT_LIMIT_DU_WI = 3

@dataclass
class Thresholds:
    urgent_cut: int = 7
    important_cut: int = 7

# --------------- SCORING ---------------
def clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(float(x))))
    except Exception:
        return default

def compute_importance(impact: int, risk_blocker: int) -> float:
    return 0.6 * impact + 0.4 * risk_blocker

def compute_quadrant(impact: int, urgency: int, risk_blocker: int, th: Thresholds) -> Tuple[str, float]:
    imp = compute_importance(impact, risk_blocker)
    is_important = imp >= th.important_cut
    is_urgent    = urgency >= th.urgent_cut
    if is_urgent and is_important:    return "DU_WI", imp
    if not is_urgent and is_important:return "ND_WI", imp
    if is_urgent and not is_important:return "DU_NWI", imp
    return "ND_NWI", imp

def katapult_score(impact: int, effort: int, urgency: int, risk_blocker: int) -> float:
    e = max(1, effort)
    return (impact / e) * (1.0 + 0.08 * urgency) * (1.0 + 0.06 * risk_blocker)

# --------------- DB ---------------
DB_PATH = os.getenv("EISENHOWER_DB", "eisenhower_app.db")

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db():
    with closing(get_conn()) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                pw_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                success_criteria TEXT NOT NULL,
                target_audience TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                primary_lever TEXT NOT NULL,
                constraints TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                quadrant TEXT NOT NULL,
                impact INTEGER NOT NULL,
                effort INTEGER NOT NULL,
                urgency INTEGER NOT NULL,
                risk_blocker INTEGER NOT NULL,
                importance REAL NOT NULL,
                rationale TEXT,
                next_action TEXT,
                dependencies TEXT,
                status TEXT NOT NULL,
                owner TEXT,
                due_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# --------------- AUTH ---------------
def create_user(email: str, pw: str) -> bool:
    email = email.strip().lower()
    if not email or len(pw) < 8:
        return False
    pw_hash = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    with closing(get_conn()) as conn:
        try:
            conn.execute(
                "INSERT INTO users(email, pw_hash, created_at) VALUES(?,?,?)",
                (email, pw_hash, now_iso()),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def verify_user(email: str, pw: str) -> Optional[int]:
    email = email.strip().lower()
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT id, pw_hash FROM users WHERE email=?", (email,)).fetchone()
        if not row:
            return None
        uid, pw_hash = row
        if bcrypt.checkpw(pw.encode(), pw_hash.encode()):
            return int(uid)
        return None

# --------------- PROJECTS ---------------
def upsert_project(user_id: int, proj: Dict[str, Any]) -> int:
    with closing(get_conn()) as conn:
        existing = conn.execute(
            "SELECT id FROM projects WHERE user_id=? AND name=?", (user_id, proj["name"])
        ).fetchone()
        t = now_iso()
        if existing:
            pid = int(existing[0])
            conn.execute(
                """UPDATE projects
                   SET success_criteria=?, target_audience=?, horizon_days=?, primary_lever=?, constraints=?, updated_at=?
                   WHERE id=?""",
                (
                    proj["success_criteria"],
                    proj["target_audience"],
                    int(proj["horizon_days"]),
                    proj["primary_lever"],
                    proj.get("constraints", ""),
                    t,
                    pid,
                ),
            )
        else:
            cur = conn.execute(
                """INSERT INTO projects(user_id, name, success_criteria, target_audience, horizon_days, primary_lever, constraints, created_at, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    user_id,
                    proj["name"],
                    proj["success_criteria"],
                    proj["target_audience"],
                    int(proj["horizon_days"]),
                    proj["primary_lever"],
                    proj.get("constraints", ""),
                    t,
                    t,
                ),
            )
            pid = int(cur.lastrowid)
        conn.commit()
        return pid

def list_projects(user_id: int) -> List[Tuple[int, str]]:
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT id, name FROM projects WHERE user_id=? ORDER BY updated_at DESC", (user_id,)
        ).fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

def load_project(user_id: int, project_id: int) -> Optional[Dict[str, Any]]:
    with closing(get_conn()) as conn:
        row = conn.execute(
            """SELECT id, name, success_criteria, target_audience, horizon_days, primary_lever, constraints
               FROM projects WHERE id=? AND user_id=?""",
            (project_id, user_id),
        ).fetchone()
        if not row:
            return None
        return {
            "id": int(row[0]),
            "name": row[1],
            "success_criteria": row[2],
            "target_audience": row[3],
            "horizon_days": int(row[4]),
            "primary_lever": row[5],
            "constraints": row[6] or "",
        }

# --------------- TASKS ---------------
def load_tasks(project_id: int) -> pd.DataFrame:
    with closing(get_conn()) as conn:
        rows = conn.execute(
            """SELECT id, title, description, quadrant, impact, effort, urgency, risk_blocker, importance,
                      rationale, next_action, dependencies, status, owner, due_date, updated_at
               FROM tasks WHERE project_id=? ORDER BY updated_at DESC""",
            (project_id,),
        ).fetchall()
    cols = [
        "id",
        "title",
        "description",
        "quadrant",
        "impact",
        "effort",
        "urgency",
        "risk_blocker",
        "importance",
        "rationale",
        "next_action",
        "dependencies",
        "status",
        "owner",
        "due_date",
        "updated_at",
    ]
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    return df

def save_tasks(project_id: int, df: pd.DataFrame):
    with closing(get_conn()) as conn:
        t = now_iso()
        for _, r in df.iterrows():
            title = str(r.get("title", "")).strip()
            if not title:
                continue
            payload = {
                "title": title,
                "description": str(r.get("description", "")).strip(),
                "quadrant": str(r.get("quadrant", "ND_WI")).strip(),
                "impact": clamp_int(r.get("impact", 5), 1, 10, 5),
                "effort": clamp_int(r.get("effort", 5), 1, 10, 5),
                "urgency": clamp_int(r.get("urgency", 5), 1, 10, 5),
                "risk_blocker": clamp_int(r.get("risk_blocker", 3), 0, 10, 3),
                "importance": float(r.get("importance", 0.0)),
                "rationale": str(r.get("rationale", "")).strip(),
                "next_action": str(r.get("next_action", "")).strip(),
                "dependencies": str(r.get("dependencies", "")).strip(),
                "status": str(r.get("status", "todo")).strip()
                if str(r.get("status", "")).strip() in STATUS_OPTIONS
                else "todo",
                "owner": str(r.get("owner", "")).strip(),
                "due_date": str(r.get("due_date", "")).strip(),
            }
            rid = r.get("id")
            is_new = (
                rid is None
                or (isinstance(rid, float) and pd.isna(rid))
                or str(rid).strip() == ""
            )
            if is_new:
                conn.execute(
                    """INSERT INTO tasks(project_id, title, description, quadrant, impact, effort, urgency, risk_blocker, importance,
                                        rationale, next_action, dependencies, status, owner, due_date, created_at, updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        project_id,
                        payload["title"],
                        payload["description"],
                        payload["quadrant"],
                        payload["impact"],
                        payload["effort"],
                        payload["urgency"],
                        payload["risk_blocker"],
                        payload["importance"],
                        payload["rationale"],
                        payload["next_action"],
                        payload["dependencies"],
                        payload["status"],
                        payload["owner"],
                        payload["due_date"],
                        t,
                        t,
                    ),
                )
            else:
                conn.execute(
                    """UPDATE tasks
                       SET title=?, description=?, quadrant=?, impact=?, effort=?, urgency=?, risk_blocker=?, importance=?,
                           rationale=?, next_action=?, dependencies=?, status=?, owner=?, due_date=?, updated_at=?
                       WHERE id=? AND project_id=?""",
                    (
                        payload["title"],
                        payload["description"],
                        payload["quadrant"],
                        payload["impact"],
                        payload["effort"],
                        payload["urgency"],
                        payload["risk_blocker"],
                        payload["importance"],
                        payload["rationale"],
                        payload["next_action"],
                        payload["dependencies"],
                        payload["status"],
                        payload["owner"],
                        payload["due_date"],
                        t,
                        int(float(rid)),
                        project_id,
                    ),
                )
        conn.commit()

def delete_task(project_id: int, task_id: int):
    with closing(get_conn()) as conn:
        conn.execute("DELETE FROM tasks WHERE id=? AND project_id=?", (task_id, project_id))
        conn.commit()

# --------------- LLM ---------------
TASK_SCHEMA_HINT = {
    "tasks": [
        {
            "title": "...",
            "description": "...",
            "impact": 1,
            "effort": 1,
            "urgency": 1,
            "risk_blocker": 0,
            "rationale": "...",
            "next_action": "...",
            "dependencies": ["..."],
        }
    ],
    "assumptions": ["..."],
    "notes": ["..."],
}

SYSTEM_PROMPT = (
    "Du bist ein extrem pragmatischer Projekt-Operator. "
    "Du lieferst keine generischen Listen. "
    "Du erzeugst Aufgaben, die eine Person sofort ausführen kann (konkrete Next Actions). "
    "Du hältst dich strikt an das JSON-Format und das Schema. "
    "Keine Markdown-Ausgabe, nur gültiges JSON."
)

def build_user_prompt(project: Dict[str, Any], min_tasks: int) -> str:
    return f"""
Projekt-Mini-Brief:
- Projektname: {project["name"]}
- Erfolgskriterium (1 Satz): {project["success_criteria"]}
- Zielgruppe: {project["target_audience"]}
- Zeithorizont: {project["horizon_days"]} Tage
- Primärer Hebel: {project["primary_lever"]}
- Constraints (Zeit/Budget/Tools): {project.get("constraints","")}

Aufgabe:
1) Erstelle mindestens {min_tasks} Aufgaben, die zur Zielerreichung notwendig sind.
2) Jede Aufgabe muss eine kurze Beschreibung + Next Action enthalten.
3) Gib Scores (1-10) für:
   - impact: Wie stark katapultiert das den Fortschritt Richtung Erfolgskriterium?
   - effort: Wie aufwendig ist es? (10=sehr aufwendig)
   - urgency: Wie zeitkritisch? (10=sofort)
   - risk_blocker: 0-10 (wie stark blockiert/riskiert es das Projekt, wenn nicht getan?)
4) Gib pro Aufgabe dependencies als Liste (kann leer sein).
5) Gib rationale (1-2 Sätze) warum impact so ist.
6) Antworte ausschließlich als gültiges JSON im Schema wie dieses Beispiel (Inhalte natürlich passend):
{json.dumps(TASK_SCHEMA_HINT, ensure_ascii=False)}
""".strip()

def extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text, flags=re.I).strip()
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return ""

def llm_generate_tasks(project: Dict[str, Any], min_tasks: int, model: str) -> Dict[str, Any]:
    if not openai_client:
        raise RuntimeError("OpenAI Client nicht verfügbar.")
    prompt = build_user_prompt(project, min_tasks)
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=4_000,
    )
    raw = response.choices[0].message.content or ""
    jstr = extract_json(raw)
    if not jstr:
        raise ValueError("Kein gültiges JSON im LLM-Response gefunden.")
    data = json.loads(jstr)
    if not isinstance(data, dict) or "tasks" not in data:
        raise ValueError("JSON passt nicht zum erwarteten Schema.")
    return data

# --------------- CACHE ---------------
@st.cache_data(ttl=3600, show_spinner=False)
def llm_generate_tasks_cached(_cache_key: str, project: Dict[str, Any], min_tasks: int, model: str) -> Dict[str, Any]:
    return llm_generate_tasks(project, min_tasks, model)

def _stable_cache_key(project: Dict[str, Any], min_tasks: int, model: str) -> str:
    payload_str = json.dumps(project, sort_keys=True) + str(min_tasks) + model
    return hashlib.md5(payload_str.encode()).hexdigest()

# --------------- UI HELPERS ---------------
def quadrant_class(q: str) -> str:
    return {
        "DU_WI": "q_du_wi",
        "ND_WI": "q_nd_wi",
        "DU_NWI": "q_du_nwi",
        "ND_NWI": "q_nd_nwi",
    }.get(q, "q_nd_wi")

def render_task_card(row: pd.Series, highlight: bool = False):
    title = str(row.get("title", "")).strip()
    status = str(row.get("status", "todo"))
    impact = int(row.get("impact", 5))
    effort = int(row.get("effort", 5))
    urgency = int(row.get("urgency", 5))
    importance = float(row.get("importance", 0.0))
    rationale = str(row.get("rationale", "")).strip()
    next_action = str(row.get("next_action", "")).strip()

    border = "2px solid rgba(120,255,160,.35)" if highlight else "1px solid rgba(255,255,255,.08)"
    st.markdown(
        f"""
<div class="tcard" style="border:{border}">
  <div class="trow">
    <div class="tleft">
      <div class="ttitle">{title}</div>
      <div class="tmeta">
        <span class="badge b_stat">Status: {status}</span>
        <span class="badge b_imp">Impact {impact}</span>
        <span class="badge b_eff">Effort {effort}</span>
        <span class="badge b_urg">Urgency {urgency}</span>
        <span class="badge">Importance {importance:.1f}</span>
      </div>
    </div>
  </div>
  {"<hr/>" if (rationale or next_action) else ""}
  {"<div class='small'><b>Warum:</b> " + rationale + "</div>" if rationale else ""}
  {"<div class='small' style='margin-top:6px;'><b>Nächster Schritt:</b> " + next_action + "</div>" if next_action else ""}
</div>
""",
        unsafe_allow_html=True,
    )

def recompute_all(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for c, default in [
        ("impact", 5),
        ("effort", 5),
        ("urgency", 5),
        ("risk_blocker", 3),
        ("quadrant", "ND_WI"),
        ("importance", 0.0),
        ("status", "todo"),
    ]:
        if c not in df.columns:
            df[c] = default
    quads, imps = [], []
    for _, r in df.iterrows():
        impact = clamp_int(r.get("impact", 5), 1, 10, 5)
        effort = clamp_int(r.get("effort", 5), 1, 10, 5)
        urgency = clamp_int(r.get("urgency", 5), 1, 10, 5)
        risk = clamp_int(r.get("risk_blocker", 3), 0, 10, 3)
        q, imp = compute_quadrant(impact, urgency, risk, th)
        quads.append(q)
        imps.append(imp)
        df.at[_, "impact"] = impact
        df.at[_, "effort"] = effort
        df.at[_, "urgency"] = urgency
        df.at[_, "risk_blocker"] = risk
    df["quadrant"] = quads
    df["importance"] = imps
    return df

def pick_one_thing(df: pd.DataFrame) -> Optional[int]:
    if df is None or df.empty:
        return None
    candidates = df[df["status"].isin(["todo", "doing"])].copy()
    if candidates.empty:
        return None
    candidates["k_score"] = candidates.apply(
        lambda r: katapult_score(
            int(r.get("impact", 5)),
            int(r.get("effort", 5)),
            int(r.get("urgency", 5)),
            int(r.get("risk_blocker", 3)),
        ),
        axis=1,
    )
    pref = {"DU_WI": 3, "ND_WI": 2, "DU_NWI": 1, "ND_NWI": 0}
    candidates["q_pref"] = candidates["quadrant"].map(pref).fillna(0)
    candidates["rank"] = candidates["k_score"] + candidates["q_pref"] * 0.8
    best = candidates.sort_values("rank", ascending=False).iloc[0]
    rid = best.get("id")
    if rid is None or (isinstance(rid, float) and pd.isna(rid)) or str(rid).strip() == "":
        return None
    try:
        return int(float(rid))
    except Exception:
        return None

# --------------- SESSION ---------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "project_id" not in st.session_state:
    st.session_state.project_id = None
if "strict_mode" not in st.session_state:
    st.session_state.strict_mode = True
if "last_llm_meta" not in st.session_state:
    st.session_state.last_llm_meta = None

# --------------- START ---------------
init_db()
st.title("🧭 Eisenhower Project Planner")
st.caption("Projekt → Aufgaben → Scores → Eisenhower → Fokus. Mit editierbarer Tabelle, Persistenz und Katapult-Task.")

# --------------- SIDEBAR ---------------
with st.sidebar:
    st.header("🔐 Login")
    if st.session_state.user_id is None:
        tab_login, tab_signup = st.tabs(["Anmelden", "Registrieren"])
        with tab_login:
            email = st.text_input("E-Mail", key="login_email")
            pw = st.text_input("Passwort", type="password", key="login_pw")
            if st.button("Anmelden", use_container_width=True):
                uid = verify_user(email, pw)
                if uid:
                    st.session_state.user_id = uid
                    st.success("Eingeloggt.")
                    st.rerun()
                else:
                    st.error("Login fehlgeschlagen.")
        with tab_signup:
            email2 = st.text_input("E-Mail", key="signup_email")
            pw2 = st.text_input("Passwort (min. 8 Zeichen)", type="password", key="signup_pw")
            if st.button("Account erstellen", use_container_width=True):
                ok = create_user(email2, pw2)
                if ok:
                    st.success("Account erstellt. Bitte anmelden.")
                else:
                    st.error("Konnte Account nicht erstellen (E-Mail evtl. schon vergeben oder Passwort zu kurz).")
    else:
        st.success("Eingeloggt ✅")
        if st.button("Ausloggen", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.project_id = None
            st.rerun()

    st.divider()
    st.header("⚙️ Regeln")
    urgent_cut = st.slider("Dringlichkeits-Schwelle", 1, 10, 7)
    important_cut = st.slider("Wichtigkeits-Schwelle (Importance)", 1, 10, 7)
    th = Thresholds(urgent_cut=urgent_cut, important_cut=important_cut)
    du_wi_limit = st.number_input(
        "Empf. Limit DO-Quadrant (DU_WI)", min_value=1, max_value=10, value=DEFAULT_LIMIT_DU_WI
    )
    st.session_state.strict_mode = st.toggle(
        "Strict Mode: Quadrant immer aus Scores",
        value=st.session_state.strict_mode,
        help="An: Quadranten werden aus Impact/Urgency/Risk berechnet. Aus: Deine manuelle Auswahl bleibt erhalten.",
    )
    st.divider()
    st.header("🤖 LLM")
    st.write("API-Key erkannt ✅" if openai_client else "API-Key fehlt ❌ (OPENAI_API_KEY setzen)")
    model = st.text_input("Model", value=OPENAI_MODEL, help="z.B. gpt-4o-mini")
    min_tasks = st.slider("Min. Aufgaben", 10, 40, 12, 1)

# --------------- GATE ---------------
if st.session_state.user_id is None:
    st.info("Bitte zuerst einloggen, damit Projekte & Tasks gespeichert werden.")
    st.stop()

# --------------- PROJECT SELECT / CREATE ---------------
projects = list_projects(st.session_state.user_id)
proj_names = ["(Neu anlegen)"] + [p[1] for p in projects]
proj_map = {p[1]: p[0] for p in projects}

colA, colB = st.columns([1.2, 2.8], gap="large")
with colA:
    st.subheader("📌 Projekt")
    choice = st.selectbox("Projekt auswählen", proj_names, index=0)
    if choice != "(Neu anlegen)":
        st.session_state.project_id = proj_map[choice]
    else:
        st.session_state.project_id = None

with colB:
    st.subheader("🧾 Mini-Brief (gegen generische Pläne)")
    if st.session_state.project_id:
        loaded = load_project(st.session_state.user_id, st.session_state.project_id) or {}
        default_name = loaded.get("name", "")
        default_success = loaded.get("success_criteria", "")
        default_aud = loaded.get("target_audience", "")
        default_h = loaded.get("horizon_days", 30)
        default_lever = loaded.get("primary_lever", "Umsatz")
        default_constraints = loaded.get("constraints", "")
    else:
        default_name = default_success = default_aud = default_constraints = ""
        default_h = 30
        default_lever = "Umsatz"

    c1, c2, c3 = st.columns([1.3, 1.0, 1.0], gap="medium")
    with c1:
        project_name = st.text_input(
            "Projektname / Hauptziel", value=default_name, placeholder="z.B. Streamlit App für Reels-Generator launchen"
        )
    with c2:
        horizon_days = st.number_input("Zeithorizont (Tage)", min_value=1, max_value=365, value=int(default_h))
    with c3:
        levers = ["Umsatz", "Reichweite", "Produkt", "Prozess", "Lernen"]
        idx = levers.index(default_lever) if default_lever in levers else 0
        primary_lever = st.selectbox("Primärer Hebel", levers, index=idx)

    success_criteria = st.text_input(
        "Erfolgskriterium (1 Satz)", value=default_success, placeholder="z.B. MVP live + 50 Nutzer in 30 Tagen"
    )
    target_audience = st.text_input(
        "Zielgruppe", value=default_aud, placeholder="z.B. Fitness-Coaches, die Reels schneller produzieren wollen"
    )
    constraints = st.text_area(
        "Constraints (optional)", value=default_constraints, height=70, placeholder="Zeit/Woche, Budget, Tools, technische Vorgaben …"
    )

    save_proj = st.button("Projekt speichern / aktualisieren", type="primary", use_container_width=True)
    if save_proj:
        if not project_name.strip() or not success_criteria.strip() or not target_audience.strip():
            st.error("Bitte Projektname, Erfolgskriterium und Zielgruppe ausfüllen.")
        else:
            proj_payload = {
                "name": project_name.strip(),
                "success_criteria": success_criteria.strip(),
                "target_audience": target_audience.strip(),
                "horizon_days": int(horizon_days),
                "primary_lever": primary_lever,
                "constraints": constraints.strip(),
            }
            pid = upsert_project(st.session_state.user_id, proj_payload)
            st.session_state.project_id = pid
            st.success("Gespeichert.")
            st.rerun()

if not st.session_state.project_id:
    st.warning("Lege ein Projekt an oder wähle eins aus, um Tasks zu verwalten.")
    st.stop()

project = load_project(st.session_state.user_id, st.session_state.project_id)
if not project:
    st.error("Projekt nicht gefunden.")
    st.stop()

st.divider()

# --------------- TASK CONTROLS ---------------
df = load_tasks(project["id"])
if not df.empty:
    for c in ["impact", "effort", "urgency", "risk_blocker"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(5).astype(int)
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
    df["status"] = df["status"].astype(str).where(df["status"].isin(STATUS_OPTIONS), "todo")

cX, cY, cZ, cW = st.columns([1.2, 1.0, 1.0, 1.2], gap="medium")
with cX:
    if st.button("🔁 Quadranten neu berechnen", use_container_width=True):
        df2 = recompute_all(df, th)
        save_tasks(project["id"], df2)
        st.success("Neu berechnet & gespeichert.")
        st.rerun()
with cY:
    if st.button("➕ Leere Aufgabe hinzufügen", use_container_width=True):
        new_row = {
            "id": "",
            "title": "Neue Aufgabe",
            "description": "",
            "quadrant": "ND_WI",
            "impact": 6,
            "effort": 3,
            "urgency": 4,
            "risk_blocker": 2,
            "importance": compute_importance(6, 2),
            "rationale": "",
            "next_action": "",
            "dependencies": "",
            "status": "todo",
            "owner": "",
            "due_date": "",
            "updated_at": now_iso(),
        }
        df2 = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
        df2 = recompute_all(df2, th)
        save_tasks(project["id"], df2)
        st.rerun()
with cZ:
    if st.button("💾 Speichern", use_container_width=True):
        df2 = recompute_all(df, th) if st.session_state.strict_mode else df.copy()
        if not df2.empty and not st.session_state.strict_mode:
            df2["importance"] = df2.apply(
                lambda r: compute_importance(
                    clamp_int(r.get("impact", 5), 1, 10, 5),
                    clamp_int(r.get("risk_blocker", 3), 0, 10, 3),
                ),
                axis=1,
            )
        save_tasks(project["id"], df2)
        st.success("Gespeichert.")
        st.rerun()
with cW:
    gen = st.button("🤖 Aufgaben generieren (LLM)", use_container_width=True, disabled=not bool(openai_client))
    if gen:
        with st.spinner("Generiere Aufgaben…"):
            try:
                cache_key = _stable_cache_key(project, int(min_tasks), model.strip())
                data = llm_generate_tasks_cached(cache_key, project, int(min_tasks), model.strip())
                tasks = data.get("tasks", [])
                assumptions = data.get("assumptions", [])
                notes = data.get("notes", [])
                st.session_state.last_llm_meta = {"assumptions": assumptions, "notes": notes}
                if not isinstance(tasks, list) or len(tasks) < 5:
                    st.error("LLM-Antwort unbrauchbar (zu wenige Tasks).")
                else:
                    rows = []
                    for t in tasks:
                        title = str(t.get("title", "")).strip()
                        if not title:
                            continue
                        impact = clamp_int(t.get("impact", 6), 1, 10, 6)
                        effort = clamp_int(t.get("effort", 4), 1, 10, 4)
                        urgency = clamp_int(t.get("urgency", 5), 1, 10, 5)
                        risk = clamp_int(t.get("risk_blocker", 3), 0, 10, 3)
                        q, imp = compute_quadrant(impact, urgency, risk, th)
                        deps = t.get("dependencies", [])
                        if isinstance(deps, list):
                            deps_s = ", ".join([str(x).strip() for x in deps if str(x).strip()])
                        else:
                            deps_s = str(deps).strip()
                        rows.append(
                            {
                                "id": "",
                                "title": title,
                                "description": str(t.get("description", "")).strip(),
                                "quadrant": q,
                                "impact": impact,
                                "effort": effort,
                                "urgency": urgency,
                                "risk_blocker": risk,
                                "importance": imp,
                                "rationale": str(t.get("rationale", "")).strip(),
                                "next_action": str(t.get("next_action", "")).strip(),
                                "dependencies": deps_s,
                                "status": "todo",
                                "owner": "",
                                "due_date": "",
                                "updated_at": now_iso(),
                            }
                        )
                    gen_df = pd.DataFrame(rows)
                    df2 = pd.concat([gen_df, df], ignore_index=True)
                    df2 = recompute_all(df2, th)
                    save_tasks(project["id"], df2)
                    st.success(f"{len(rows)} Aufgaben hinzugefügt.")
                    st.rerun()
            except Exception as e:
                st.error(f"Generierung fehlgeschlagen: {e}")

with st.expander("🧠 Hinweise/Annahmen der letzten Generierung", expanded=False):
    meta = st.session_state.last_llm_meta or {}
    assumptions = meta.get("assumptions", [])
    notes = meta.get("notes", [])
    if not assumptions and not notes:
        st.caption("Noch keine Generierung in dieser Session oder keine Meta-Daten vorhanden.")
    else:
        if assumptions:
            st.markdown("**Annahmen**")
            for a in assumptions:
                st.write(f"- {a}")
        if notes:
            st.markdown("**Notizen**")
            for n in notes:
                st.write(f"- {n}")

st.divider()

# --------------- BOARD VIEW ---------------
df = load_tasks(project["id"])
df = recompute_all(df, th) if st.session_state.strict_mode else df
one_id = pick_one_thing(df) if not df.empty else None

du_wi_count = int((df["quadrant"] == "DU_WI").sum()) if not df.empty else 0
if du_wi_count > int(du_wi_limit):
    st.warning(
        f"Du hast {du_wi_count} Tasks in **Dringend & Wichtig**. Empfehlung: max. {int(du_wi_limit)}. "
        "Alles darüber ist meist Überforderung → schiebe Dinge nach PLAN/DELEGIEREN oder zerlege."
    )

st.subheader("🗂️ Editierbare Task-Tabelle (Single Source of Truth)")
if df.empty:
    st.info("Noch keine Tasks. Generiere welche oder füge manuell hinzu.")
else:
    editor_df = df.copy()
    q_label = {k: QUADRANTS[k] for k in QUADRANTS}
    q_rev = {v: k for k, v in q_label.items()}
    editor_df["quadrant_label"] = editor_df["quadrant"].map(q_label)
    display_cols = [
        "id",
        "title",
        "status",
        "quadrant_label",
        "impact",
        "effort",
        "urgency",
        "risk_blocker",
        "importance",
        "next_action",
        "dependencies",
        "rationale",
        "owner",
        "due_date",
    ]
    edited = st.data_editor(
        editor_df[display_cols],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "id": st.column_config.TextColumn("ID", disabled=True, width="small"),
            "title": st.column_config.TextColumn("Titel", required=True, width="large"),
            "status": st.column_config.SelectboxColumn("Status", options=STATUS_OPTIONS, width="small"),
            "quadrant_label": st.column_config.SelectboxColumn("Eisenhower", options=list(q_label.values()), width="medium"),
            "impact": st.column_config.NumberColumn("Impact", min_value=1, max_value=10, step=1),
            "effort": st.column_config.NumberColumn("Effort", min_value=1, max_value=10, step=1),
            "urgency": st.column_config.NumberColumn("Urgency", min_value=1, max_value=10, step=1),
            "risk_blocker": st.column_config.NumberColumn("Risk/Blocker", min_value=0, max_value=10, step=1),
            "importance": st.column_config.NumberColumn("Importance", disabled=True),
            "next_action": st.column_config.TextColumn("Nächster Schritt", width="large"),
            "dependencies": st.column_config.TextColumn("Abhängigkeiten", width="large"),
            "rationale": st.column_config.TextColumn("Warum (Impact-Begründung)", width="large"),
            "owner": st.column_config.TextColumn("Owner", width="small"),
            "due_date": st.column_config.TextColumn("Fällig (frei)", width="small"),
        },
        key="task_editor",
    )
    c1, c2 = st.columns([1.2, 2.8], gap="medium")
    with c1:
        if st.button("✅ Änderungen übernehmen", use_container_width=True):
            edited2 = edited.copy()
            edited2["quadrant"] = edited2["quadrant_label"].map(q_rev).fillna("ND_WI")
            edited2.drop(columns=["quadrant_label"], inplace=True, errors="ignore")
            if st.session_state.strict_mode:
                edited2 = recompute_all(edited2, th)
            else:
                edited2["importance"] = edited2.apply(
                    lambda r: compute_importance(
                        clamp_int(r.get("impact", 5), 1, 10, 5),
                        clamp_int(r.get("risk_blocker", 3), 0, 10, 3),
                    ),
                    axis=1,
                )
            save_tasks(project["id"], edited2)
            st.success("Änderungen gespeichert.")
            st.rerun()
    with c2:
        st.caption(
            "Strict Mode an = Quadrant wird immer berechnet (konsistent). "
            "Strict Mode aus = deine Auswahl bleibt, Importance wird trotzdem berechnet."
        )

st.divider()

# --------------- QUADRANT BOARD ---------------
st.subheader("🟦 Eisenhower Board (farblich + Fokus)")
q_dfs = {q: df[df["quadrant"] == q].copy() for q in QUADRANTS.keys()}
for q, qdf in q_dfs.items():
    if qdf.empty:
        continue
    qdf["k_score"] = qdf.apply(
        lambda r: katapult_score(
            int(r.get("impact", 5)),
            int(r.get("effort", 5)),
            int(r.get("urgency", 5)),
            int(r.get("risk_blocker", 3)),
        ),
        axis=1,
    )
    status_rank = {"todo": 0, "doing": 1, "blocked": 2, "done": 3}
    qdf["status_rank"] = qdf["status"].map(status_rank).fillna(9)
    q_dfs[q] = qdf.sort_values(["status_rank", "k_score"], ascending=[True, False])

cA, cB = st.columns(2, gap="large")
cC, cD = st.columns(2, gap="large")
quad_layout = [
    ("DU_WI", cA, "🔥 DO"),
    ("ND_WI", cB, "🧠 PLAN"),
    ("DU_NWI", cC, "⚡ DELEGIEREN"),
    ("ND_NWI", cD, "🧹 ELIMINIEREN"),
]
for q, col, tag in quad_layout:
    with col:
        st.markdown(
            f"<div class='qbox {quadrant_class(q)}'>"
            f"<div class='qhdr'>{tag} — {QUADRANTS[q]}</div>"
            f"<div class='qsub'>Sortiert nach Katapult-Potenzial (Impact/Effort + Urgency + Blocker).</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        qdf = q_dfs[q]
        if qdf.empty:
            st.caption("— leer —")
        else:
            if q == "DU_WI" and len(qdf) > int(du_wi_limit):
                st.warning("Zu viele DO-Tasks. Schiebe runter oder zerlege, bis max. Limit erreicht.")
            for _, row in qdf.iterrows():
                highlight = (
                    one_id is not None
                    and str(row.get("id", "")).strip() != ""
                    and int(float(row["id"])) == one_id
                )
                render_task_card(row, highlight=highlight)

st.divider()

# --------------- ONE THING ---------------
st.subheader("🎯 Katapult-Task (One Thing)")
if one_id is None:
    st.info("Kein aktiver Task gefunden (alles done?).")
else:
    one_row = df[df["id"].astype(str) == str(one_id)]
    if one_row.empty:
        st.info("Etwas ist schief gelaufen – Task nicht gefunden.")
    else:
        render_task_card(one_row.iloc[0], highlight=True)
st.caption("Regel: Diese eine Sache zuerst. Wenn du nur 30 Minuten hast — genau daran arbeiten.")

# --------------- DELETE ---------------
with st.expander("🧨 Danger Zone (Task löschen)", expanded=False):
    st.write("Löschen ist endgültig.")
    del_id = st.text_input("Task-ID zum Löschen", placeholder="z.B. 12")
    if st.button("Task löschen"):
        try:
            tid = int(float(del_id.strip()))
            if tid <= 0:
                raise ValueError
            delete_task(project["id"], tid)
            st.success("Gelöscht.")
            st.rerun()
        except (ValueError, IndexError):
            st.error("Bitte gültige numerische Task-ID eingeben.")

st.divider()
st.caption("Hinweis: Passwort-Hashing erfolgt sicher per bcrypt – produktionsbereit.")
