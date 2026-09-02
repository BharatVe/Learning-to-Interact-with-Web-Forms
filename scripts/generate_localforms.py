#!/usr/bin/env python3
"""Recreate the 50 form specs as a locally hosted FormFactory-style Flask site.

FormFactory (arXiv:2506.01520, https://github.com/formfactory-ai/formfactory) hosts
its benchmark forms as Flask-rendered HTML templates (Bootstrap 4 form-group markup,
one route per form, gold answers as JSON). This script reproduces that pattern for
the 50 forms already defined in src/forms/*/spec.json, so the same content can be
served from a second, non-Google-Forms platform.

Two additive, documented adaptations keep the recreation compatible with this
repo's existing verifier/scoring contract (see docs/ALTERNATIVE_PLATFORM_PLAN.md):
  1. Each question is wrapped in a Google-Forms-compatible `role="listitem"`
     container (the verifier locates questions this way regardless of platform).
  2. Each radio/checkbox input carries an explicit `role="radio"`/`role="checkbox"`
     and `aria-label` so the same role-based selectors used for Google Forms can
     read selection state from native HTML controls.

Everything else (Bootstrap classes, form-group/mb-3 wrapper divs, label/input
structure, one Flask route per form, JSON submission logging) mirrors upstream
FormFactory directly.

Outputs (all regenerated deterministically from src/forms/*/spec.json):
  - evaluation_additions/formfactory_import/site/templates/<form_id>.html
  - evaluation_additions/formfactory_import/site/app.py
  - src/forms_localforms/lf_<form_id>/spec.json
  - data/answers_localforms/lf_<form_id>/runs.json
"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
FORMS_SRC = ROOT_DIR / "src" / "forms"
ANSWERS_SRC = ROOT_DIR / "data" / "answers"
SITE_DIR = ROOT_DIR / "evaluation_additions" / "formfactory_import" / "site"
TEMPLATES_DIR = SITE_DIR / "templates"
FORMS_OUT = ROOT_DIR / "src" / "forms_localforms"
ANSWERS_OUT = ROOT_DIR / "data" / "answers_localforms"

LF_PREFIX = "lf_"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "field"


def field_identity(form_id: str, q_order: int, q_title: str) -> str:
    return f"{form_id}_{q_order}_{slugify(q_title)}"


def esc(text: str) -> str:
    return html.escape(str(text or ""), quote=True)


def render_question(form_id: str, q: dict) -> str:
    q_order = q["q_order"]
    q_title = q["q_title"]
    q_type = q["q_type"]
    required = bool(q.get("required"))
    help_text = q.get("help_text") or ""
    field_id = field_identity(form_id, q_order, q_title)
    req_attr = ' required aria-required="true"' if required else ""
    req_marker = ' <span class="text-danger">*</span>' if required else ""

    format_hint = ""
    if q_type == "DATE":
        format_hint = "Format: YYYY-MM-DD"
    elif q_type == "TIME":
        format_hint = "Format: HH:MM (24-hour)"
    if format_hint:
        help_text = f"{help_text} ({format_hint})".strip() if help_text else format_hint

    parts = [f'  <div class="mb-3" role="listitem" data-question-order="{q_order}">']
    parts.append(f'    <label class="form-label" for="{field_id}">{esc(q_title)}{req_marker}</label>')
    if help_text:
        parts.append(f'    <div class="form-text">{esc(help_text)}</div>')

    if q_type == "SHORT_TEXT":
        parts.append(
            f'    <input type="text" class="form-control" id="{field_id}" name="{field_id}"{req_attr}>'
        )
    elif q_type == "PARAGRAPH":
        parts.append(
            f'    <textarea class="form-control" id="{field_id}" name="{field_id}" rows="4"{req_attr}></textarea>'
        )
    elif q_type == "SINGLE_CHOICE":
        options = [o.strip() for o in str(q.get("options") or "").split(";") if o.strip()]
        for i, opt in enumerate(options):
            opt_id = f"{field_id}_{i}"
            parts.append('    <div class="form-check">')
            parts.append(
                f'      <input class="form-check-input" type="radio" role="radio" '
                f'name="{field_id}" id="{opt_id}" value="{esc(opt)}" aria-label="{esc(opt)}"{req_attr}>'
            )
            parts.append(f'      <label class="form-check-label" for="{opt_id}">{esc(opt)}</label>')
            parts.append("    </div>")
    elif q_type == "MULTI_CHOICE":
        options = [o.strip() for o in str(q.get("options") or "").split(";") if o.strip()]
        for i, opt in enumerate(options):
            opt_id = f"{field_id}_{i}"
            parts.append('    <div class="form-check">')
            parts.append(
                f'      <input class="form-check-input" type="checkbox" role="checkbox" '
                f'name="{field_id}" id="{opt_id}" value="{esc(opt)}" aria-label="{esc(opt)}">'
            )
            parts.append(f'      <label class="form-check-label" for="{opt_id}">{esc(opt)}</label>')
            parts.append("    </div>")
    elif q_type == "DROPDOWN":
        options = [o.strip() for o in str(q.get("options") or "").split(";") if o.strip()]
        parts.append(f'    <select class="form-select" id="{field_id}" name="{field_id}"{req_attr}>')
        parts.append('      <option value="" disabled selected>Select an option</option>')
        for opt in options:
            parts.append(f'      <option value="{esc(opt)}">{esc(opt)}</option>')
        parts.append("    </select>")
    elif q_type == "DATE":
        parts.append(
            f'    <input type="text" class="form-control" id="{field_id}" name="{field_id}" '
            f'placeholder="YYYY-MM-DD"{req_attr}>'
        )
    elif q_type == "TIME":
        parts.append(
            f'    <input type="text" class="form-control" id="{field_id}" name="{field_id}" '
            f'placeholder="HH:MM"{req_attr}>'
        )
    else:
        raise ValueError(f"Unsupported q_type {q_type!r} in form {form_id}")

    parts.append("  </div>")
    return "\n".join(parts)


TEMPLATE_SKELETON = """{{% extends 'base.html' %}}
{{% block title %}}{title}{{% endblock %}}
{{% block content %}}
<div class="card shadow">
  <div class="card-body">
    <h1 class="mb-2">{title}</h1>
    <p class="text-muted">{description}</p>
    <form method="POST" id="{form_id}Form">
      <div role="list">
{questions}
      </div>
      <button type="submit" class="btn btn-primary mt-3">Submit</button>
    </form>
  </div>
</div>
{{% endblock %}}
"""

BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}LocalForms{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/bootstrap.min.css') }}">
</head>
<body>
    <div class="container mt-5 mb-5">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

HOME_TEMPLATE = """{% extends 'base.html' %}
{% block title %}LocalForms{% endblock %}
{% block content %}
<h1 class="mb-4">LocalForms</h1>
<p class="text-muted">FormFactory-style recreation of the 50-form dataset, served as native-HTML forms.</p>
<ul class="list-group">
{% for form_id, title in forms %}
  <li class="list-group-item"><a href="{{ url_for('render_form', form_id=form_id) }}">{{ title }}</a></li>
{% endfor %}
</ul>
{% endblock %}
"""

SUBMITTED_TEMPLATE = """{% extends 'base.html' %}
{% block title %}Submitted{% endblock %}
{% block content %}
<div class="card shadow">
  <div class="card-body text-center">
    <h1 class="mb-3">Your response has been recorded</h1>
    <p class="text-muted">{{ title }}</p>
  </div>
</div>
{% endblock %}
"""

APP_PY_TEMPLATE = '''"""FormFactory-style Flask site serving the recreated 50-form dataset.

Generated by scripts/generate_localforms.py — do not edit by hand.
Modeled directly on https://github.com/formfactory-ai/formfactory app.py
(one route per form, POST submissions logged to submission/<form_id>.json).
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

FORM_TITLES = {form_titles}


@app.route("/")
def home():
    forms = sorted(FORM_TITLES.items(), key=lambda kv: kv[0])
    return render_template("home.html", forms=forms)


@app.route("/forms/<form_id>", methods=["GET", "POST"])
def render_form(form_id):
    if form_id not in FORM_TITLES:
        return jsonify({{"error": "unknown form_id", "form_id": form_id}}), 404
    if request.method == "POST":
        data = request.form.to_dict(flat=False)
        save_submission_to_json(form_id, data)
        # A rendered HTML confirmation page, not a bare JSON response: a raw
        # `application/json` response is shown by Chrome's built-in JSON
        # viewer, whose text is not reliably exposed through body.innerText()
        # the way normal rendered HTML text is (confirmed by comparing an
        # accessibility snapshot, which did see the confirmation text, against
        # innerText, which did not, for the same page). This repo's scoring
        # harness detects a successful submission by reading body.innerText()
        # for a confirmation phrase, so the confirmation must be real HTML.
        return render_template("submitted.html", title=FORM_TITLES[form_id])
    return render_template(f"{{form_id}}.html")


def save_submission_to_json(form_id, data):
    submission_dir = BASE_DIR / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    path = submission_dir / f"{{form_id}}.json"
    entries = []
    if path.exists():
        entries = json.loads(path.read_text(encoding="utf-8"))
    data = dict(data)
    data["submission_time"] = datetime.now(timezone.utc).isoformat()
    entries.append(data)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("LOCALFORMS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LOCALFORMS_PORT", "5000")))
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
'''


def main() -> None:
    if TEMPLATES_DIR.exists():
        shutil.rmtree(TEMPLATES_DIR)
    TEMPLATES_DIR.mkdir(parents=True)
    (SITE_DIR / "static" / "css").mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "submission").mkdir(parents=True, exist_ok=True)

    if FORMS_OUT.exists():
        shutil.rmtree(FORMS_OUT)
    FORMS_OUT.mkdir(parents=True)
    if ANSWERS_OUT.exists():
        shutil.rmtree(ANSWERS_OUT)
    ANSWERS_OUT.mkdir(parents=True)

    (TEMPLATES_DIR / "base.html").write_text(BASE_HTML, encoding="utf-8")
    (TEMPLATES_DIR / "home.html").write_text(HOME_TEMPLATE, encoding="utf-8")
    (TEMPLATES_DIR / "submitted.html").write_text(SUBMITTED_TEMPLATE, encoding="utf-8")

    form_titles = {}
    form_ids = sorted(p.name for p in FORMS_SRC.iterdir() if p.is_dir() and (p / "spec.json").exists())

    for form_id in form_ids:
        spec = json.loads((FORMS_SRC / form_id / "spec.json").read_text(encoding="utf-8"))
        lf_id = f"{LF_PREFIX}{form_id}"
        title = spec.get("form_title") or form_id
        description = spec.get("form_description") or ""

        questions_html = "\n".join(render_question(form_id, q) for q in spec["questions"])
        template_html = TEMPLATE_SKELETON.format(
            title=esc(title),
            description=esc(description),
            form_id=form_id,
            questions=questions_html,
        )
        (TEMPLATES_DIR / f"{lf_id}.html").write_text(template_html, encoding="utf-8")
        form_titles[lf_id] = title

        lf_spec = dict(spec)
        lf_spec["form_id"] = lf_id
        lf_spec["form_title"] = title
        lf_spec["form_description"] = description
        lf_spec["source"] = "generate_localforms.py recreation of src/forms/%s/spec.json (FormFactory-style)" % form_id
        lf_spec["edit_url"] = None
        lf_spec["published_url"] = None
        lf_spec["form_url"] = f"http://127.0.0.1:5000/forms/{lf_id}"
        questions = []
        for q in spec["questions"]:
            q2 = dict(q)
            q2["q_title"] = q["q_title"]
            questions.append(q2)
        lf_spec["questions"] = questions

        out_dir = FORMS_OUT / lf_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "spec.json").write_text(json.dumps(lf_spec, ensure_ascii=False, indent=2), encoding="utf-8")

        answers_path = ANSWERS_SRC / form_id / "runs.json"
        answers_doc = json.loads(answers_path.read_text(encoding="utf-8"))
        answers_doc["form_id"] = lf_id
        answers_doc["description"] = (
            f"{answers_doc.get('description', '')} (recreated for LocalForms platform comparison)".strip()
        )
        out_answers_dir = ANSWERS_OUT / lf_id
        out_answers_dir.mkdir(parents=True, exist_ok=True)
        (out_answers_dir / "runs.json").write_text(
            json.dumps(answers_doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    app_py = APP_PY_TEMPLATE.format(form_titles=json.dumps(form_titles, ensure_ascii=False, indent=4))
    (SITE_DIR / "app.py").write_text(app_py, encoding="utf-8")

    print(f"[OK] generated {len(form_ids)} forms")
    print(f"[OK] site: {SITE_DIR}")
    print(f"[OK] specs: {FORMS_OUT}")
    print(f"[OK] answers: {ANSWERS_OUT}")


if __name__ == "__main__":
    main()
