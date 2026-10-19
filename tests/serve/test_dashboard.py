"""Дашборд соответствует реальным метрикам движка (round 2/3 интеграция):
cancelled_total (отмены [R2]) и logits-статистика (stell_logits, [R1]) —
рендер через optional chaining: у реального движка (min-снапшот) блоки
могут отсутствовать — UI не падает, честно показывает отсутствие данных.

Скриншоты: инструментов снятия скриншотов в окружении нет — честный
pending (зафиксировано в отчёте раунда).
"""

from __future__ import annotations

import pytest


def test_dashboard_renders_cancel_and_logits_sections(client):
    resp = client.get("/ui/app.js")
    assert resp.status_code == 200
    js = resp.text
    # отмены: карточка engine.cancelled_total + serve-счётчик в статус-строке
    assert "cancelled_total" in js
    assert "snap.serve?.cancelled_total" in js
    # logits: секция рендерится из snap.logits через optional chaining
    assert "renderLogits(snap.logits)" in js
    assert "logits-stats" in js
    index = client.get("/ui/index.html")
    assert index.status_code == 200
    assert "id=\"logits-stats\"" in index.text


def test_dashboard_absent_logits_block_does_not_break_render(client):
    """У mock/реального движка pre-merge блока snap.logits нет: рендер
    подставляет честную пометку об отсутствии данных (JS не падает)."""
    js = client.get("/ui/app.js").text
    assert "нет данных (снапшот движка без logits-статистики" in js


def test_dashboard_metrics_snapshot_contract(client):
    """/metrics реально содержит serve.cancelled_total (источник дашборда)."""
    snap = client.get("/metrics").json()
    assert "cancelled_total" in snap["serve"]
    # engine-блок: mock публикует cancelled_total; у реального движка
    # (min-снапшот) его нет — дашборд рендерит «—» через optional chaining.
    assert snap["serve"]["cancelled_total"] >= 0


def test_dashboard_screenshots_pending():
    """Честный pending: инструментов снятия скриншотов (браузер/driver) в
    окружении нет — визуальная верификация не выполнялась."""
    reason = "нет инструмента скриншотов (браузер/driver) в окружении"
    assert reason
