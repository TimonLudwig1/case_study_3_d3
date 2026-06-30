# Session-Log — Aufbau der CashLog Case Study (`caseStudy.py`)

Dieses Dokument protokolliert **exakt**, was in dieser Claude-Code-Session passiert ist:
jeder Arbeitsschritt, jedes Problem, jeder Fix und jede begründete Entscheidung – in
chronologischer Reihenfolge. Es ergänzt die im Notebook selbst dokumentierten Begründungen.

---

## 0. Ausgangslage

- **Auftrag (aus `CLAUDE.md`):** Für CashLog das optimale Cash-Center-Netzwerk bestimmen
  (welche der 42 Standorte behalten/schließen) und das Vorgehen in **einem** ausführbaren
  Notebook nachvollziehbar dokumentieren. Oberste Regel: *jeder Schritt begründet & verteidigt*.
- **Vorhandene Dateien zu Beginn:** nur `CLAUDE.md` und `Data-driven_SCM_CashLog.md`.

---

## 1. Kontext aufbauen

1. `CLAUDE.md` vollständig gelesen (war bereits im Kontext).
2. `Data-driven_SCM_CashLog.md` vollständig gelesen → Verständnis von Problem, PPDAC,
   Warehouse-Location-Modell, Transportkostenformel, erweitertem Modell (5 Typen),
   Sensitivitätsanalyse.
3. Projektverzeichnis inspiziert.

**Verständnis der Aufgabe (Soll-Ergebnis):**
1. Transportkosten `c_ij = 480 · demand · minPerStop / (450 − 2·travelTime)` herleiten,
   berechnen, gegen `shifts_with_costs.csv` validieren.
2. Basis-Warehouse-Location-Modell lösen + Karte + Plausibilisierung.
3. Erweitertes Modell (5 Typen, `Σ_t y_it ≤ 1`, stückweise lineare Kosten).
4. Sensitivitätsanalyse (Nachfrage ↓, Schichtkosten ↑, neue Technologie).
5. Robuste Empfehlung (behalten/schließen/kritisch) + Grenzen.

---

## 2. Problem #1 — Fehlende Dateien

**Problem:** `CLAUDE.md` verweist auf `data/…` CSVs und auf `basicAnalysis_week_1.py`.
Beides war **nicht** im Projekt vorhanden (`data/` leer, kein Notebook).

**Vorgehen:**
- Mit `find` bestätigt, dass nur die zwei Markdown-Dateien existierten.
- Den User **nicht stillschweigend umgangen**, sondern den Blocker klar benannt und per
  Rückfrage geklärt (zweimal `AskUserQuestion` gestellt; der User wollte beide Male zuerst
  selbst etwas klären – also nachgefragt statt Annahmen zu treffen).

**Lösung durch den User:**
- Die Daten liegen in einem öffentlichen GitHub-Repo
  (`D3IP-SS25/data-driven-scm-dataset`), das Basis-Notebook lädt sie direkt per Raw-URL.
- Der User hat anschließend `basicAnalysis_week_1.py` ins Projekt gelegt.

---

## 3. Daten beschaffen & prüfen

1. Vier CSVs per `curl` heruntergeladen (nach `data/`):
   `warehouses.csv` (42), `regions.csv` (515), `shifts.csv` (21.630 = 42×515),
   `shifts_with_costs.csv` (21.630, Benchmark).
2. Header & Zeilenzahlen geprüft → Struktur stimmt mit den Folien überein.
3. Bestätigt: in `shifts.csv` ist `transportationCosts == travelTime` (Platzhalter, wie
   angekündigt).
4. Per GitHub-API geprüft, ob das Basis-Notebook im Repo liegt → **nein**, nur Daten.
   Deshalb war es wichtig, dass der User das Notebook separat liefert.

---

## 4. Bestehenden Code verstehen (`basicAnalysis_week_1.py`)

Vollständig gelesen und die Konventionen dokumentiert:

| Element | Im Basis-Notebook |
|---|---|
| Format | **marimo** (v0.23.9) |
| Modell-Bibliothek | **PuLP** (`pulp as pl`) |
| Solver | **HiGHS** (`pl.HiGHS`) |
| Daten | GitHub-Raw-URLs, indiziert nach IDs |
| Variablen | `x_ij` (binär), `y_i` (binär) |
| Zielfunktion | `Σ x_ij c_ij + Σ y_i f_i` |
| Constraints | `x_ij ≤ y_i`, `Σ_i x_ij = 1` |
| Karte | **folium** |

**Wichtige Beobachtung:** Das Basis-Notebook rechnet mit dem **Platzhalter**
`transportationCosts`. Die korrekten Kosten zu berechnen ist genau Teil 1.

**Entscheidungen (begründet):**
- **Format = marimo** (`caseStudy.py`) → gleiches Ökosystem, kein Stilbruch.
- **Solver/Bibliothek = PuLP + HiGHS** → identisch zum Basis-Notebook.

---

## 5. Vorab-Validierung der Kernrisiken (vor dem Bauen)

Statt „blind" das Notebook zu schreiben, wurden die **risikoreichsten Korrektheitspunkte**
zuerst in kleinen Skripten geprüft:

### 5.1 Transportkostenformel vs. Benchmark
- **Erstversuch hatte einen Vergleichsfehler:** `mine` und `ref` wurden positionell
  verglichen, obwohl die **Zeilenreihenfolge der beiden CSVs unterschiedlich** ist
  (`shifts.csv` beginnt mit `(1,74)`, `shifts_with_costs.csv` mit `(45,475)`).
- **Fix:** `ref.reindex(shifts.index)` → Reihenfolge angleichen.
- **Ergebnis:** Formel exakt korrekt — max. relative Abweichung **4,3·10⁻¹⁶** auf allen
  2.661 erreichbaren Verbindungen.

### 5.2 Behandlung unerreichbarer Verbindungen
- **Entdeckung:** 18.969 von 21.630 Links (87,7 %) haben `2·travelTime ≥ 450` →
  in **einer** Schicht unerreichbar.
- Die Referenzdatei setzt diese auf einen **Big-M** = `480 · 9.999.999 ≈ 4,8·10⁹`.
- **Entscheidung:** Im Optimierungsmodell solche Links **ausschließen** (keine `x`-Variable),
  in der **Validierung** aber den Big-M exakt reproduzieren (100 % Übereinstimmung).

### 5.3 Bleibt das Modell bei Ausschluss zulässig?
- Geprüft: **jede** der 515 Regionen hat ≥1 erreichbares Center (min 1, median 5, max 12).
- ⇒ Ausschluss ist sicher; Regionen mit nur 1 Option erzwingen „ihr" Center → wichtiger
  Hinweis für die Robustheits-Diskussion.

### 5.4 Basismodell als Kalibrierung
- Basismodell schnell gelöst: **optimal, 18 Center, 98,6 Mio €** (Fix 32,4M / Transport 66,3M),
  Center-Volumina 37k…717k.
- Diese Volumina dienten zur **Kalibrierung der 5 Center-Typ-Grenzen** in Teil 3.

---

## 6. Notebook bauen (`caseStudy.py`)

Vollständiges marimo-Notebook geschrieben mit allen fünf Teilen, jeweils Markdown-Begründung
davor und Interpretation danach. Struktur:

- Konfigurationszelle (alle Konstanten benannt + Quelle): `SHIFT_MIN=450`, `SHIFT_COST=480`,
  Big-M, Szenario-Faktoren, 5 Center-Typen.
- Teil 1: Herleitung + `link_costs(...)`-Funktion (wiederverwendbar für Szenarien) + Validierung
  (Kennzahlen + Log-Log-Scatter).
- Teil 2: `solve_base(...)` + Ergebnis + Constraint-Validierung + folium-Karte + Plausibilisierung.
- Teil 3: `solve_extended(...)` (5 Typen, `Σ_t y_it ≤ 1`) + Vergleich Basis vs. erweitert.
- Teil 4: alle Szenarien über `solve_base` + Übersichtstabelle + Heatmap + Robustheits-Klassifikation.
- Teil 5: Management-Empfehlung + Grenzen & Annahmen.

**Bewusste Abweichung vom Basis-Notebook (begründet):** kein interaktiver „Solve"-Button,
sondern **eager** rechnen → das Notebook läuft garantiert von oben nach unten durch
(Reproduzierbarkeit, CLAUDE.md §4).

---

## 7. Verifikation & die Performance-Probleme

Headless getestet via `marimo export script caseStudy.py` → flaches Skript ausführen.

### 7.1 Erster End-to-End-Lauf
- Lief **fehlerfrei durch (Exit 0)**.

### 7.2 Problem #2 — Erweitertes MILP zu langsam / hängt
- Ein **separater** Test des erweiterten MILP **timeoutete bei 2 min**, obwohl der volle
  Notebook-Lauf (im Hintergrund, ohne Kill) durchgelaufen war → die Lösung war **grenzwertig
  langsam (~90–120 s) und unzuverlässig**.
- **Diagnose:**
  - `pl.HiGHS` akzeptiert `timeLimit`/`gapRel` (per `inspect` geprüft) — der frühere „Timeout"
    war nur der Bash-2-min-Kill, nicht HiGHS.
  - Mit `msg=True` analysiert: HiGHS erreichte nach 25 s nur eine **Optimalitätslücke von 2,3 %**.
  - HiGHS warnte explizit: **Zielfunktion skalieren** (Kosten ~10⁸).
  - Ursache: 2.661 binäre `x` + 210 binäre `y`, große, nicht-konvexe stückweise lineare
    Kostenkurve, schwache LP-Relaxation durch große `z ≤ V_ub · y`.

### 7.3 Fixes (iterativ getestet)
1. **Engere `h`-Obergrenze:** Max. von einem einzelnen Center erreichbares Volumen berechnet
   = **907.412**. `V_ub(h)` von 3.000.000 → **950.000** (gültige, enge Schranke).
   → Lücke nach 70 s: 1,18 %. Besser, aber noch zu langsam.
2. **Zielfunktion skalieren (÷1000):** in Tausend € rechnen (HiGHS-Empfehlung).
   → Lücke 0,5 % nach 95 s. Numerik deutlich besser.
3. **`x` zu kontinuierlich relaxieren (Test):** Lösungszeit fiel auf **21,6 s**, ABER
   **1,2 % der `x` wurden fraktional** → Single-Sourcing verletzt (Regionen aufgeteilt),
   nicht folien-konform. → **verworfen**.
4. **Finale Entscheidung:** `x` **binär lassen** (korrekt & folientreu) + **Skalierung ÷1000**
   + **`gapRel=0.01` (1 %)** + **`timeLimit=180`** als Sicherheitsnetz.
   - Die 1 %-Lücke wird **ehrlich berichtet** (kein „beweisbar optimal"-Overclaim); für eine
     strategische Behalten-/Schließen-Entscheidung ist sie unkritisch, die Center-/Typ-Struktur
     ist stabil.

### 7.4 Ergänzungen nach den Fixes
- Konfigurationszelle um `OBJ_SCALE`, `EXT_GAP_REL`, `EXT_TIME_LIMIT` erweitert.
- `solve_extended` skaliert die Zielfunktion und skaliert das Ergebnis zurück; liefert Status,
  Volumen je Center und Typ.
- **Volumen-Check** ergänzt: jedes verarbeitete Volumen liegt im `[V_lb, V_ub]` seines Typs.
- **Constraint-Validierungszelle** fürs Basismodell ergänzt (alle Regionen bedient, genau eine
  Zuordnung, nur erreichbare Links, nur offene Center).

---

## 8. Finale Verifikation (alle Zahlen bestätigt)

End-to-End-Lauf nach allen Fixes: **Exit 0 in 58 s**. Headline-Zahlen separat nachgerechnet:

| Prüfung | Ergebnis |
|---|---|
| Teil 1 — Formel vs. Benchmark | max. rel. Abw. **4,3·10⁻¹⁶**, Big-M-Match **100 %**, 2.661 erreichbare Links |
| Teil 2 — Basismodell | **Optimal**, **18** Center, 98,6 Mio €, Constraints ✅ |
| Teil 4 — Nachfrage −10/−30/−50 % | 18 → 18 → 17 → 16 Center (**weniger**, wie erwartet) |
| Teil 4 — Schichtkosten +20/+50 % | 18 → 19 → 19 Center (**mehr/dezentraler**, wie erwartet) |
| Teil 4 — autonom / autonom+24·7 | 15 → **5** Center (**starke Konsolidierung**, wie erwartet) |
| Robustheit | 2 immer offen · 20 nie offen · 20 szenarioabhängig |

**Befund:** Alle drei Sensitivitätsachsen bewegen sich in die **theoretisch erwartete Richtung**
→ das Modell verhält sich ökonomisch plausibel.

---

## 9. Offene Punkte / bewusste Annahmen

- **Center-Typ-Parameter (Teil 3)** (`V_lb/ub`, `c_fix`, `c_var`) sind dokumentierte, kalibrierte
  **Eigen-Setzungen** (Skaleneffekte), da die Folien keine konkreten Zahlen liefern. Sie stehen
  oben in der Konfigurationszelle und sind leicht änderbar — bei vorgegebenen Kurswerten dort
  ersetzen.
- **1 % Optimalitätslücke** im erweiterten Modell (bewusst, begründet).
- **`data/`-Ordner:** lokal heruntergeladene Kopie zur Inspektion/Offline-Fallback. Das Notebook
  liest – wie das Basis-Notebook – aus den GitHub-Raw-URLs; `data/` ist also redundant und kann
  gelöscht werden.
- **Robustes „immer offen" = nur 2 Center**, weil das extreme Szenario *autonom + 24/7* das Netz
  auf 5 Center kollabieren lässt. Das ist ein ehrlicher Befund, kein Fehler; Teil 5 rahmt es als
  behalten / schließen / „kritisch (Optionen offenhalten)".

---

## 10. Erzeugte/geänderte Dateien

- **`caseStudy.py`** — das vollständige marimo-Notebook (Teile 1–5). *(neu)*
- **`data/`** — lokale Kopie der vier CSVs. *(neu, optional/redundant)*
- **`SESSION_LOG.md`** — dieses Protokoll. *(neu)*

### Notebook starten
```bash
marimo edit caseStudy.py    # interaktiv bearbeiten
marimo run caseStudy.py     # als App ausführen
```
Hinweis: Der erweiterte-Modell-Lauf dauert ~30–90 s (Lösung bis 1 % Lücke); benötigt
Internet (Daten von GitHub).
