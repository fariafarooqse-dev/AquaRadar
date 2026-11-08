import sqlite3

import pytest

import database as db
from model_service import AnalysisResult


def result(status="completed", cls="leak", conf=0.8):
    return AnalysisResult(status, cls if status == "completed" else None,
                          conf if status != "error" else None,
                          "leak_model_v1" if status != "error" else None,
                          "msg", ["warn1", "warn2"])


def path(tmp_path):
    return tmp_path / "t.db"


def test_save_and_read_analysis_preserves_result(tmp_path):
    p = path(tmp_path)
    aid = db.save_analysis(result(), "a.wav", " Lab ", "  note ", db_path=p)
    row = db.get_analysis(aid, p)
    assert row["status"] == "completed" and row["predicted_class"] == "leak"
    assert row["confidence"] == 0.8 and row["model_version"] == "leak_model_v1"
    assert row["location_label"] == "Lab" and row["notes"] == "note"
    assert row["warnings"] == '["warn1", "warn2"]'


def test_error_analysis_is_stored_without_fake_values(tmp_path):
    p = path(tmp_path)
    aid = db.save_analysis(result("error"), "bad.wav", db_path=p)
    row = db.get_analysis(aid, p)
    assert row["status"] == "error" and row["predicted_class"] is None and row["confidence"] is None


def test_alert_requires_location_and_reason(tmp_path):
    p = path(tmp_path)
    with pytest.raises(ValueError):
        db.create_alert("  ", "reason", db_path=p)
    with pytest.raises(ValueError):
        db.create_alert("Lab", "", db_path=p)


def test_alert_cannot_come_from_failed_or_missing_analysis(tmp_path):
    p = path(tmp_path)
    bad = db.save_analysis(result("error"), "bad.wav", db_path=p)
    with pytest.raises(ValueError):
        db.create_alert("Lab", "r", analysis_id=bad, db_path=p)
    with pytest.raises(ValueError):
        db.create_alert("Lab", "r", analysis_id=999, db_path=p)


def test_alert_links_analysis_and_leaves_it_unchanged(tmp_path):
    p = path(tmp_path)
    aid = db.save_analysis(result(), "a.wav", db_path=p)
    before = db.get_analysis(aid, p)
    alert = db.create_alert("Lab", "Potential leak-like audio", analysis_id=aid, db_path=p)
    db.update_alert_status(alert, "Under Review", db_path=p)
    db.update_alert_status(alert, "Resolved", "False alarm", "checked on site", db_path=p)
    assert db.get_analysis(aid, p) == before
    a = db.get_alert(alert, p)
    assert a["status"] == "Resolved" and a["inspection_outcome"] == "False alarm"
    assert "checked on site" in a["notes"]


def test_manual_alert_without_analysis(tmp_path):
    p = path(tmp_path)
    alert = db.create_alert("Library", "Operator heard hissing", db_path=p)
    rows = db.list_alerts(db_path=p)
    assert rows[0]["id"] == alert and rows[0]["analysis_id"] is None and rows[0]["predicted_class"] is None


def test_status_rules(tmp_path):
    p = path(tmp_path)
    alert = db.create_alert("Lab", "r", db_path=p)
    with pytest.raises(ValueError):
        db.update_alert_status(alert, "Resolved", db_path=p)                     # outcome required
    with pytest.raises(ValueError):
        db.update_alert_status(alert, "Under Review", "False alarm", db_path=p)  # outcome only on resolve
    with pytest.raises(ValueError):
        db.update_alert_status(alert, "Nonsense", db_path=p)
    with pytest.raises(ValueError):
        db.update_alert_status(alert, "New", db_path=p)                          # not forward
    db.update_alert_status(alert, "Inspection Required", db_path=p)              # skipping ahead allowed
    with pytest.raises(ValueError):
        db.update_alert_status(alert, "Under Review", db_path=p)                 # backwards
    db.update_alert_status(alert, "Resolved", "Leak found and repaired", db_path=p)
    with pytest.raises(ValueError):
        db.update_alert_status(alert, "Resolved", "Inconclusive", db_path=p)     # closed
    assert db.get_alert(alert, p)["inspection_outcome"] == "Leak found and repaired"


def test_unknown_alert_raises(tmp_path):
    with pytest.raises(ValueError):
        db.update_alert_status(42, "Under Review", db_path=path(tmp_path))


def test_summary_counts_come_from_saved_rows(tmp_path):
    p = path(tmp_path)
    assert db.summary_counts(p)["analyses_total"] == 0
    a1 = db.save_analysis(result(), "a.wav", db_path=p)
    db.save_analysis(result("error"), "b.wav", db_path=p)
    db.save_analysis(result("uncertain"), "c.wav", db_path=p)
    al = db.create_alert("Lab", "r", analysis_id=a1, db_path=p)
    db.create_alert("Lib", "r2", db_path=p)
    db.update_alert_status(al, "Resolved", "Inconclusive", db_path=p)
    s = db.summary_counts(p)
    assert s["analyses_total"] == 3
    assert s["analyses_by_status"] == {"completed": 1, "error": 1, "uncertain": 1}
    assert s["alerts_total"] == 2 and s["alerts_open"] == 1
    assert s["alerts_by_status"]["Resolved"] == 1 and s["alerts_by_status"]["New"] == 1


def test_old_starter_schema_is_migrated(tmp_path):
    p = path(tmp_path)
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE analyses (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL "
                 "DEFAULT CURRENT_TIMESTAMP, filename TEXT NOT NULL, location_label TEXT, status TEXT NOT NULL, "
                 "predicted_class TEXT, confidence REAL, model_version TEXT, message TEXT)")
    conn.execute("CREATE TABLE alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id INTEGER, created_at TEXT "
                 "NOT NULL DEFAULT CURRENT_TIMESTAMP, location_label TEXT, reason TEXT NOT NULL, status TEXT NOT "
                 "NULL DEFAULT 'New', inspection_outcome TEXT, notes TEXT)")
    conn.commit()
    conn.close()
    aid = db.save_analysis(result(), "a.wav", db_path=p)
    alert = db.create_alert("Lab", "r", analysis_id=aid, db_path=p)
    assert db.get_alert(alert, p)["updated_at"] is not None
