import json
import os
import sqlite3
from functools import wraps

import joblib
import pandas as pd
from flask import (Flask, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key =os.environ.get("SECRET_KEY", "dev-secret-key")
DB = "users.db"
FEATURES = ["Temperature", "Humidity", "WindSpeed"]
LIMITS = {"Temperature": (-20, 60), "Humidity": (0, 100), "WindSpeed": (0, 150)}
ICONS = {"Sunny": "☀️", "Cloudy": "☁️", "Rainy": "🌧️"}

model = joblib.load("model/weather_model.pkl")
metrics = json.load(open("model/metrics.json"))
tree_rules = open("model/tree_rules.txt").read()


# ---------- database ----------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    db = g.pop("db", None)
    if db:
        db.close()


def init_db():
    db = sqlite3.connect(DB)
    db.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        temperature REAL, humidity REAL, windspeed REAL,
        result TEXT, confidence REAL,
        created TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """)
    db.commit()
    db.close()


def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*a, **kw)
    return wrapper


# ---------- auth ----------
@app.route("/")
def index():
    return redirect(url_for("dashboard" if "user_id" in session else "login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        pw = request.form["password"]
        if not name or not email or len(pw) < 6:
            flash("Fill all fields. Password must be at least 6 characters.", "error")
        else:
            try:
                get_db().execute("INSERT INTO users(name,email,password) VALUES(?,?,?)",
                                 (name, email, generate_password_hash(pw)))
                get_db().commit()
                flash("Account created! Please login.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That email is already registered.", "error")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        user = get_db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password"], request.form["password"]):
            session["user_id"], session["name"] = user["id"], user["name"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- dashboard ----------
@app.route("/dashboard")
@login_required
def dashboard():
    rows = get_db().execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 8",
        (session["user_id"],)).fetchall()
    return render_template("dashboard.html", m=metrics, rows=rows,
                           icons=ICONS, rules=tree_rules)


@app.route("/history")
@login_required
def history():
    rows = get_db().execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC",
        (session["user_id"],)).fetchall()
    return render_template("history.html", rows=rows, icons=ICONS)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json(silent=True) or {}
    vals = {}
    for f in FEATURES:
        try:
            v = float(data.get(f))
        except (TypeError, ValueError):
            return jsonify(error=f"{f} must be a number."), 400
        lo, hi = LIMITS[f]
        if not lo <= v <= hi:
            return jsonify(error=f"{f} must be between {lo} and {hi}."), 400
        vals[f] = v

    X = pd.DataFrame([vals], columns=FEATURES)
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    conf = round(float(proba.max()) * 100, 1)
    probs = {c: round(float(p) * 100, 1) for c, p in zip(model.classes_, proba)}

    get_db().execute(
        "INSERT INTO predictions(user_id,temperature,humidity,windspeed,result,confidence) "
        "VALUES(?,?,?,?,?,?)",
        (session["user_id"], vals["Temperature"], vals["Humidity"], vals["WindSpeed"], label, conf))
    get_db().commit()
    return jsonify(label=label, icon=ICONS.get(label, ""), confidence=conf, probabilities=probs)


init_db()

if __name__ == "__main__":
    app.run(debug=True)
