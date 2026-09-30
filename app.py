"""CardioCheck - Heart Disease Risk Prediction web application.

Run:  python app.py   ->  http://127.0.0.1:5000
"""
import csv
import io
import json
import sqlite3
from datetime import datetime
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from flask import (Flask, Response, abort, jsonify, redirect, render_template, request, send_file, url_for)

from config import (CATEGORICAL, DB_PATH, load_data, FEATURE_META, FEATURES, MODEL_DIR, NUMERIC, TARGET, risk_band)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB uploads

DISCLAIMER = ("This tool is an educational machine-learning project and is NOT a medical device. "
              "It cannot diagnose heart disease. Please consult a qualified doctor.")


# --------------------------------------------------------------------------- model helpers
def _file(name):
    return MODEL_DIR / (name.replace(" ", "_").replace("(", "").replace(")", "") + ".joblib")


def ensure_models():
    """Load the saved models; if they are missing or were built with an incompatible
    scikit-learn/numpy version, retrain them automatically on this machine."""
    try:
        meta = json.loads((MODEL_DIR / "metadata.json").read_text())
        for n in meta["results"]:
            joblib.load(_file(n))
    except Exception as e:  # noqa
        print(f"[CardioCheck] Saved models unusable ({e!r}). Retraining - this takes ~1 minute...")
        import train
        train.main()
        metadata.cache_clear()
        load_model.cache_clear()


@lru_cache(maxsize=None)
def load_model(name):
    return joblib.load(_file(name))


@lru_cache(maxsize=1)
def metadata():
    path = MODEL_DIR / "metadata.json"
    if not path.exists():
        raise RuntimeError("Models not found. Run `python train.py` first.")
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def dataset():
    return load_data()


def model_names():
    return list(metadata()["results"].keys())


def validate(payload):
    """Validate & coerce a dict of raw inputs. Returns (clean_dict, errors)."""
    clean, errors = {}, {}
    for f in FEATURES:
        meta = FEATURE_META[f]
        raw = payload.get(f)
        if raw in (None, ""):
            errors[f] = "Required"
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors[f] = "Must be a number"
            continue
        if meta["type"] == "select":
            if int(val) not in meta["options"]:
                errors[f] = "Invalid choice"
                continue
            val = int(val)
        else:
            if not meta["min"] <= val <= meta["max"]:
                errors[f] = f"Must be between {meta['min']} and {meta['max']}"
                continue
            if f != "oldpeak":
                val = int(round(val))
        clean[f] = val
    return clean, errors


def predict_proba(clean, model_name=None):
    model_name = model_name or metadata()["best_model"]
    df = pd.DataFrame([clean])[FEATURES]
    return float(load_model(model_name).predict_proba(df)[0, 1])


def explain(clean, model_name):
    """Model-agnostic local explanation by feature 'neutralisation'.

    For each feature we replace the patient's value with the typical (median / mode)
    value from the training data and measure how the risk changes.
    Positive = this value pushes risk UP, negative = pushes risk DOWN.
    """
    meta = metadata()
    model = load_model(model_name)
    base = float(model.predict_proba(pd.DataFrame([clean])[FEATURES])[0, 1])
    rows = []
    variants = []
    for f in FEATURES:
        v = dict(clean)
        v[f] = meta["medians"][f] if f in NUMERIC else meta["modes"][f]
        variants.append(v)
    probs = model.predict_proba(pd.DataFrame(variants)[FEATURES])[:, 1]
    for f, p in zip(FEATURES, probs):
        rows.append({"feature": f, "label": FEATURE_META[f]["label"],
                     "value": display_value(f, clean[f]), "impact": round(base - float(p), 4)})
    rows.sort(key=lambda r: -abs(r["impact"]))
    return rows


def display_value(f, v):
    m = FEATURE_META[f]
    if m["type"] == "select":
        return m["options"][int(v)]
    return f"{v:g} {m.get('unit', '')}".strip()


def advice(clean, prob):
    """Rule based lifestyle tips derived from the entered values."""
    tips = []
    if clean["trestbps"] >= 140:
        tips.append(("Blood pressure", "Resting BP is in the hypertensive range. Reduce salt, stay active and monitor regularly."))
    elif clean["trestbps"] >= 120:
        tips.append(("Blood pressure", "Slightly elevated BP. Lifestyle changes can help keep it in check."))
    if clean["chol"] >= 240:
        tips.append(("Cholesterol", "High cholesterol. Cut saturated/trans fats, eat more fibre, discuss a lipid panel with your doctor."))
    elif clean["chol"] >= 200:
        tips.append(("Cholesterol", "Borderline cholesterol. Prefer whole grains, nuts and fish."))
    if clean["fbs"] == 1:
        tips.append(("Blood sugar", "Elevated fasting glucose - get screened for diabetes and limit refined sugars."))
    if clean["exang"] == 1:
        tips.append(("Exercise angina", "Chest pain on exertion should be medically evaluated before intense exercise."))
    max_hr = 220 - clean["age"]
    if clean["thalach"] < 0.7 * max_hr:
        tips.append(("Heart rate", f"Peak heart rate ({clean['thalach']} bpm) is low for your age (estimated max {max_hr}). Ask about stress-test follow-up."))
    if clean["oldpeak"] >= 2:
        tips.append(("ST depression", "Significant ST depression noted - it is an important marker to review with a cardiologist."))
    if prob >= 0.6:
        tips.insert(0, ("Next step", "Book an appointment with a cardiologist soon for a clinical evaluation."))
    elif prob >= 0.3:
        tips.insert(0, ("Next step", "Consider a routine cardiac check-up and discuss these values with your doctor."))
    tips.append(("General", "Aim for 150 min/week of moderate activity, a Mediterranean-style diet, no smoking, 7-8h sleep."))
    return tips


def population_context(clean):
    """Percentile of the patient's numeric values within the dataset (for radar/percentile UI)."""
    df = dataset()
    out = {}
    for f in NUMERIC:
        out[f] = round(float((df[f] < clean[f]).mean() * 100), 1)
    return out


# --------------------------------------------------------------------------- database
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, patient TEXT, model TEXT,
            probability REAL, risk TEXT, inputs TEXT)""")


def save_prediction(patient, model_name, prob, risk, clean):
    with db() as conn:
        cur = conn.execute("INSERT INTO predictions(created,patient,model,probability,risk,inputs) VALUES(?,?,?,?,?,?)",
                           (datetime.now().strftime("%Y-%m-%d %H:%M"), patient or "Anonymous", model_name,
                            prob, risk, json.dumps(clean)))
        return cur.lastrowid


def full_result(clean, model_name, patient=None, save=True):
    prob = predict_proba(clean, model_name)
    risk, color = risk_band(prob)
    result = {"probability": round(prob, 4), "percent": round(prob * 100, 1), "risk": risk, "color": color,
              "prediction": int(prob >= 0.5), "model": model_name,
              "label": "Heart disease likely" if prob >= 0.5 else "Heart disease unlikely",
              "explanation": explain(clean, model_name), "advice": advice(clean, prob),
              "percentiles": population_context(clean),
              "consensus": {n: round(predict_proba(clean, n), 4) for n in model_names()}}
    if save:
        result["id"] = save_prediction(patient, model_name, prob, risk, clean)
    return result


# --------------------------------------------------------------------------- pages
@app.context_processor
def inject():
    return {"disclaimer": DISCLAIMER, "year": datetime.now().year}


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True, "invert_target": metadata().get("invert_target", False)})


@lru_cache(maxsize=1)
def sample_patients():
    """Real rows from the dataset with the lowest / highest predicted risk (for the demo buttons)."""
    df = dataset()
    probs = load_model(metadata()["best_model"]).predict_proba(df[FEATURES])[:, 1]
    lo, hi = df.iloc[int(np.argmin(probs))], df.iloc[int(np.argmax(probs))]
    conv = lambda r: {f: (float(r[f]) if f == "oldpeak" else int(r[f])) for f in FEATURES}
    return {"low": conv(lo), "high": conv(hi)}


@app.route("/")
def index():
    return render_template("index.html", features=FEATURES, meta=FEATURE_META, models=model_names(),
                           best=metadata()["best_model"], samples=sample_patients())


@app.route("/predict", methods=["POST"])
def predict():
    """AJAX endpoint used by the form."""
    payload = request.get_json(silent=True) or request.form.to_dict()
    clean, errors = validate(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    model_name = payload.get("model") if payload.get("model") in model_names() else metadata()["best_model"]
    result = full_result(clean, model_name, payload.get("patient"))
    result["inputs"] = clean
    return jsonify({"ok": True, **result})


@app.route("/whatif", methods=["POST"])
def whatif():
    """Predict without saving - used by the what-if sliders."""
    payload = request.get_json(silent=True) or {}
    clean, errors = validate(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    model_name = payload.get("model") if payload.get("model") in model_names() else metadata()["best_model"]
    prob = predict_proba(clean, model_name)
    risk, color = risk_band(prob)
    return jsonify({"ok": True, "percent": round(prob * 100, 1), "risk": risk, "color": color})


@app.route("/dashboard")
def dashboard():
    df = dataset()
    stats = {
        "total": len(df), "disease": int(df[TARGET].sum()),
        "avg_age": round(float(df.age.mean()), 1), "avg_chol": round(float(df.chol.mean()), 1),
    }
    charts = {
        "target": [int((df[TARGET] == 0).sum()), int((df[TARGET] == 1).sum())],
        "age_bins": _age_bins(df),
        "cp": _by_cat(df, "cp"), "sex": _by_cat(df, "sex"), "exang": _by_cat(df, "exang"),
        "thal": _by_cat(df, "thal"), "ca": _by_cat(df, "ca"), "slope": _by_cat(df, "slope"),
        "corr": _corr(df),
        "scatter": [{"x": int(r.age), "y": int(r.thalach), "t": int(r.target)} for r in df.itertuples()],
    }
    return render_template("dashboard.html", stats=stats, charts=charts, meta=FEATURE_META,
                           importance=metadata()["importance"])


def _age_bins(df):
    bins = [0, 40, 50, 60, 70, 120]
    labels = ["<40", "40-49", "50-59", "60-69", "70+"]
    g = pd.cut(df.age, bins=bins, labels=labels, right=False)
    return {"labels": labels,
            "healthy": [int(((g == l) & (df[TARGET] == 0)).sum()) for l in labels],
            "disease": [int(((g == l) & (df[TARGET] == 1)).sum()) for l in labels]}


def _by_cat(df, col):
    opts = FEATURE_META[col]["options"]
    keys = sorted(int(k) for k in opts)
    return {"labels": [opts[k] for k in keys],
            "healthy": [int(((df[col] == k) & (df[TARGET] == 0)).sum()) for k in keys],
            "disease": [int(((df[col] == k) & (df[TARGET] == 1)).sum()) for k in keys]}


def _corr(df):
    c = df.corr().round(2)
    return {"labels": list(c.columns), "matrix": c.values.tolist()}


@app.route("/models")
def models_page():
    return render_template("models.html", meta=metadata())


@app.route("/history")
def history():
    with db() as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 200").fetchall()
    rows = [dict(r, color=risk_band(r["probability"])[1]) for r in rows]
    return render_template("history.html", rows=rows)


@app.route("/history/<int:pid>/delete", methods=["POST"])
def delete_record(pid):
    with db() as conn:
        conn.execute("DELETE FROM predictions WHERE id=?", (pid,))
    return redirect(url_for("history"))


@app.route("/history/export.csv")
def export_history():
    with db() as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "created", "patient", "model", "probability", "risk"] + FEATURES)
    for r in rows:
        inp = json.loads(r["inputs"])
        w.writerow([r["id"], r["created"], r["patient"], r["model"], r["probability"], r["risk"]] +
                   [inp.get(f) for f in FEATURES])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=predictions.csv"})


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", features=FEATURES, models=model_names(), best=metadata()["best_model"],
                               results=None, error=None)
    file = request.files.get("file")
    model_name = request.form.get("model")
    model_name = model_name if model_name in model_names() else metadata()["best_model"]
    error = None
    results = None
    try:
        df = pd.read_csv(file)
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        if len(df) > 2000:
            raise ValueError("Maximum 2000 rows per upload.")
        X = df[FEATURES].apply(pd.to_numeric, errors="raise")
        if X.isna().any().any():
            raise ValueError("File contains empty cells in required columns.")
        probs = load_model(model_name).predict_proba(X)[:, 1]
        out = df.copy()
        out["risk_probability_%"] = (probs * 100).round(1)
        out["risk_level"] = [risk_band(p)[0] for p in probs]
        out["prediction"] = (probs >= 0.5).astype(int)
        app.config["LAST_BATCH"] = out.to_csv(index=False)
        summary = out["risk_level"].value_counts().to_dict()
        results = {"rows": out.head(200).to_dict("records"), "cols": list(out.columns), "n": len(out),
                   "summary": summary, "model": model_name}
    except Exception as e:  # noqa
        error = str(e)
    return render_template("batch.html", features=FEATURES, models=model_names(), best=model_name,
                           results=results, error=error)


@app.route("/batch/download")
def batch_download():
    data = app.config.get("LAST_BATCH")
    if not data:
        abort(404)
    return Response(data, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=batch_predictions.csv"})


@app.route("/batch/template.csv")
def batch_template():
    df = dataset().drop(columns=[TARGET]).sample(8, random_state=1)
    return Response(df.to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=template.csv"})


@app.route("/report/<int:pid>.pdf")
def report(pid):
    from report import build_pdf
    with db() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id=?", (pid,)).fetchone()
    if not row:
        abort(404)
    clean = json.loads(row["inputs"])
    result = full_result(clean, row["model"], save=False)
    pdf = build_pdf(dict(row), clean, result)
    return send_file(io.BytesIO(pdf), mimetype="application/pdf", download_name=f"heart_report_{pid}.pdf")


@app.route("/about")
def about():
    return render_template("about.html", meta=metadata())


# --------------------------------------------------------------------------- REST API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API. Body: the 13 features (+ optional "model"). Returns probability & risk."""
    payload = request.get_json(silent=True) or {}
    clean, errors = validate(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    model_name = payload.get("model") if payload.get("model") in model_names() else metadata()["best_model"]
    prob = predict_proba(clean, model_name)
    risk, _ = risk_band(prob)
    return jsonify({"ok": True, "model": model_name, "probability": round(prob, 4), "risk": risk,
                    "prediction": int(prob >= 0.5)})


@app.route("/api/models")
def api_models():
    return jsonify(metadata()["results"])


@app.errorhandler(Exception)
def on_error(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    app.logger.exception("Unhandled error")
    if request.path.startswith(("/predict", "/whatif", "/api")):
        return jsonify({"ok": False, "errors": {"server": f"{type(e).__name__}: {e}"}}), 500
    return render_template("about.html", meta=metadata()), 500


@app.errorhandler(413)
def too_large(_):
    return render_template("batch.html", features=FEATURES, models=model_names(), best=metadata()["best_model"],
                           results=None, error="File too large (max 2 MB)."), 413


init_db()
ensure_models()

if __name__ == "__main__":
    # debug/reloader off by default: the auto-reloader can kill in-flight requests
    # (shows up in the browser as "Failed to fetch"). Set FLASK_DEBUG=1 to enable it.
    import os
    app.run(host="127.0.0.1", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1",
            use_reloader=False, threaded=True)
