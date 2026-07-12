# Data-driven Supply Chain Management – Die CashLog-Fallstudie

> Begleitende Erklärung zu den Vorlesungsfolien „Data Driven Decisions (D3) in Practice –
> Data-driven Supply Chain Management" (Lehrstuhl für Logistik & quantitative Methoden /
> Wirtschaftsinformatik & KI, Universität Würzburg, Pibernik / Gust).

Dieses Dokument fasst die Folien zusammen und erklärt die dahinterliegenden Konzepte so,
dass du den roten Faden, die Mathematik und die praktischen Stolpersteine wirklich verstehst.

---

## 1. Worum geht es? (Lernziele)

Die Fallstudie zeigt an einem **realen Problem der Distributionsnetzwerk-Optimierung**, wie
man Optimierungsmodelle in der Praxis tatsächlich einsetzt. Der Lernfokus liegt nicht auf dem
„schönen" Modell allein, sondern auf dem ganzen Weg dorthin:

- ein reales Problem im Netzwerkdesign verstehen,
- die **Schwierigkeiten** beim Einsatz von Optimierungsmodellen in der echten Welt erkennen,
- **Daten sammeln und aufbereiten**, um die nötigen Modellparameter zu gewinnen,
- ein Optimierungsmodell an die reale Entscheidung **anpassen**,
- aus den Ergebnissen **sinnvolle, belastbare Erkenntnisse** ableiten,
- die **Grenzen** des Modells erkennen und überwinden.

Die zentrale Botschaft: In der Praxis steckt der größte Aufwand selten im Lösen des Modells,
sondern im *Problem richtig fassen* und in der *Datenaufbereitung*.

---

## 2. Der Arbeitsrahmen: PPDAC

Die gesamte Fallstudie ist entlang des **PPDAC-Zyklus** strukturiert – einem allgemeinen
Vorgehensmodell für datengetriebene Entscheidungen:

| Phase | Bedeutung | Leitfrage |
|-------|-----------|-----------|
| **P**roblem | Problem präzise definieren | Welches Problem lösen wir genau? |
| **P**lan | Vorgehen festlegen | Wie gehen wir es an, welche Daten/Analysen brauchen wir? |
| **D**ata | Daten beschaffen & aufbereiten | Welche Daten liegen vor, wie machen wir sie nutzbar? |
| **A**nalysis | Analysieren & modellieren | Was sagt das Modell, was die Sensitivitätsanalyse? |
| **C**onclusion | Schlussfolgern & empfehlen | Welche Empfehlung geben wir – wie robust ist sie? |

PPDAC ist ein **Zyklus**: Erkenntnisse aus der Analyse führen oft zurück zu einer schärferen
Problemdefinition oder zu neuen Datenanforderungen.

---

## 3. PROBLEM – Das CashLog-Netzwerk

### Was macht CashLog?

CashLog ist der **spanische Marktführer im Cash-Logistik-Geschäft**. Cash-Logistik umfasst:

1. **Abholung** von Bargeld bei Supermärkten, Banken, Tankstellen usw.
2. **Zählen, Sortieren, Lagern** des Bargelds in hochsicheren „Cash Centern".
3. **Verteilung** von Bargeld an Banken und andere Kunden.

Historisch betrieb CashLog ein Cash Center in **jeder Hauptstadt jeder der 50 spanischen
Provinzen**. Grund: die Nähe zur Zentralbank, die in jeder Hauptstadt eine Filiale hatte. Bei
einem Ungleichgewicht konnte CashLog dort kurzfristig Bargeld abgeben oder beziehen.

**Heute:** 42 Cash Center, ca. **42.000 Kundenstandorte** (spanisches Festland).

### Warum muss das Netz neu gestaltet werden?

Mehrere Entwicklungen haben die ursprüngliche Logik (ein Center pro Provinzhauptstadt)
überholt:

- Die **Zentralbank** hat die meisten Filialen geschlossen, und Nähe zur Zentralbank ist nicht
  mehr nötig: Cash Center haben einen eigenen Zentralbank-Tresor, und Zentralbankgeld ist
  landesweit innerhalb von **2 Stunden** beschaffbar.
- Die **Bargeldmenge sinkt** – elektronische Zahlungen verdrängen Bargeld.
- Cash Center sind **sehr teuer** (Sicherheit, Überwachung, Versicherung).

> **Hauptfrage:** *Wie soll das Netz der Cash Center in Zukunft aussehen?*

**Wichtige Einschränkung:** CashLog will **keine neuen Standorte eröffnen**. Es geht also nur
um die Auswahl: *Welche der 42 bestehenden Standorte behalten, welche schließen?*

### Den Problemkern prüfen

Bevor man rechnet, lohnen sich Kontrollfragen (Folie „Relevant Questions"):

- Welches Problem lösen wir? → Standortauswahl unter Kostengesichtspunkten.
- Was ist die *präzise* Frage? → Welche Teilmenge der 42 Center minimiert die Gesamtkosten,
  während alle Kunden bedient werden?
- Verstehe ich das Problem? Kann ich es einem Fünfjährigen erklären? → „Wir haben 42 Lager und
  müssen herausfinden, welche wir zumachen können, ohne dass es teurer wird, alle Läden zu
  beliefern."
- Ist es mit den verfügbaren Daten lösbar? → Genau das prüft die Data-Phase.

---

## 4. PLAN – Modellierung & ökonomische Logik

### Aufwärm-Überlegungen

- **Welche Parameter treiben die Entscheidung?** Fixkosten je Center, Transport-/Bedienkosten
  zu den Kunden, Nachfragevolumen, Fahrzeiten/Entfernungen, Kapazitäten.
- **Ökonomisches Ziel in einem Satz:** *Minimiere die gesamten jährlichen Kosten (Fixkosten der
  betriebenen Center + Transportkosten zur Bedienung aller Kunden), sodass die gesamte Nachfrage
  bedient wird.*
- **Grundlegender Trade-off:** Mehr offene Center → höhere Fixkosten, aber **kürzere Wege**
  (niedrigere Transportkosten). Weniger Center → niedrigere Fixkosten, aber **längere Wege**
  (höhere Transportkosten). Das Optimum liegt dazwischen.
- **Warum kann man Standorte nicht isoliert bewerten? → „Netzwerkeffekte".** Ob sich ein
  Standort lohnt, hängt davon ab, *welche anderen Standorte offen sind*. Schließt man Center A,
  müssen dessen Kunden von B, C … übernommen werden – das verändert deren Auslastung und Kosten.
  Die Standorte sind also **voneinander abhängig**; man muss das Netz **als Ganzes** optimieren.

### Das klassische Warehouse-Location-Modell

Die Vorlage ist das **Warehouse Location Problem** (Standortplanungsmodell):

$$\min \; C(X,Y) = \sum_i \sum_j x_{ij}\, c_{ij} \;+\; \sum_i f_i\, y_i$$

unter den Nebenbedingungen:

| # | Nebenbedingung | Bedeutung |
|---|----------------|-----------|
| (1) | $\sum_j x_{ij} \le cap_i\, y_i \quad \forall i$ | Ein Center kann nur liefern, wenn es offen ist ($y_i=1$), und nur bis zur Kapazität $cap_i$. |
| (2) | $\sum_i x_{ij} = d_j \quad \forall j$ | Die Nachfrage jedes Kunden $j$ muss vollständig gedeckt werden. |
| (3) | $x_{ij} \ge 0 \quad \forall i,j$ | Liefermengen nicht-negativ. |
| (4) | $y_i \in \{0,1\} \quad \forall i$ | Standort offen (1) oder geschlossen (0). |

**Notation:**
- $i$ = Standort (Cash Center / Warehouse), $j$ = Kunde bzw. Kundenregion.
- $x_{ij}$ = Zuordnung/Liefermenge von $i$ nach $j$.
- $c_{ij}$ = Transportkosten von $i$ nach $j$.
- $f_i$ = jährliche Fixkosten, falls Center $i$ betrieben wird.
- $y_i$ = Öffnungsentscheidung.

**Logik:** Die Zielfunktion bildet genau den Trade-off ab – der erste Term sind die
Transportkosten, der zweite die Fixkosten der offenen Center. Das Modell wählt automatisch die
kostenoptimale Kombination offener Standorte **und** die Kundenzuordnung.

### Warum passt das Modell nicht 1:1 auf CashLog?

Genau das ist die didaktische Pointe der Folie „Which problems do you see?". Drei reale
Schwierigkeiten:

1. **Was ist $c_{ij}$ wirklich?** Es gibt keine simplen „Kosten pro Lieferung". CashLog bedient
   viele Kunden in einer 8-Stunden-Schicht über **Touren** (Routing), nicht in Einzelfahrten.
   → siehe Problem 1.
2. **42.000 Kunden** sind zu viele Einzelpunkte → man muss zu **Regionen aggregieren**
   (Clustering). → Datenvorverarbeitung.
3. **Kapazität & Fixkosten sind nicht konstant.** Cash Center können vergrößert/verkleinert
   werden; die Kosten hängen vom Volumen ab (Skaleneffekte). → Problem 3.

Die Consulting-Aufgabe besteht darin, ein überzeugendes Vorgehen zu entwerfen, das die **Logik**
des Warehouse-Location-Modells übernimmt, aber an diese Realitäten anpasst.

---

## 5. DATA – Verfügbare Daten & Aufbereitung

### Rohdaten: Kunden & Lieferungen

CashLog liefert eine Datenbank mit allen **42.000 Kunden**: Kundentyp (ATM/Retail/Branch),
Kundencode, Postleitzahl und – monatlich aufgeschlüsselt – die **Anzahl der Servicevorgänge pro
Jahr** (wie oft Geld geliefert/abgeholt wurde). Diese Anzahl ist das relevante **Volumen** (Zahl
der „Stops"), nicht der Geldbetrag.

### Warum Vorverarbeitung nötig ist (Clustering, Teil 1)

Man könnte $c_{ij}$ direkt aus Kundendaten berechnen – **aber nur, wenn jeder Kunde einzeln**
bedient würde (Hin- und Rückfahrt pro Kunde). Bei CashLog ist das nicht so:

- Jeder LKW bedient in einer **8-Stunden-Schicht viele Kunden** auf einer Tour.
- Jeden Morgen wird ein **Routing-Problem** gelöst (welche Kunden, wie viele LKW + 3er-Crews).
- Die Touren hängen von der **Tagesnachfrage** und der Center-Zuordnung ab und variieren täglich.

Das **strategische** Netzwerkdesign und das **operative** Routing lassen sich nicht in einem Zug
gemeinsam lösen. Um die Transportkosten exakt zu erfassen, müsste man für **jede** mögliche
Teilmenge geschlossener Center ein integriertes Standort-Routing-Problem über ein ganzes Jahr
lösen. Die Zahl der Optionen ist astronomisch:

$$\sum_{k=1}^{M} \binom{M}{k} = \sum_{k=1}^{M} \frac{M!}{k!\,(M-k)!} = 2^{42}-1 \approx 4{,}398 \times 10^{12}$$

(rund **4,4 Billionen** Kombinationen für $M=42$). Das ist praktisch unmöglich.

**Lösungsidee:** Kunden zu **Kundenregionen clustern**. Jede Region wird *einem* Center
zugeordnet, und das Routing wird je Region separat (näherungsweise) gelöst.

**Anforderung an eine gute Region:**
- **groß genug** im Volumen, damit man an jedem Tag effiziente Touren bilden kann,
- **klein genug**, damit es „optimal" ist, eine ganze Region einem einzigen Center zuzuordnen.

### Wie geclustert wurde (Clustering, Teil 2)

- Kunden über **Postleitzahlen** aggregieren → ca. **4.200 PLZ mit Kunden**.
- Geokoordinaten (lat/lon) jeder PLZ bestimmen.
- (Euklidische) **Distanzen** zwischen allen PLZ berechnen.
- **Clustering-Algorithmus** anwenden (z. B. *Hierarchical Agglomerative Clustering*) für
  verschiedene Clusterzahlen $N$. Leitidee: Distanzen *innerhalb* eines Clusters minimieren,
  Distanzen *zwischen* Clustern maximieren.
- Volumen je Cluster prüfen und das **kleinste $N$** wählen, das in jedem Cluster noch genug
  Volumen liefert.

Das Clustering ist in der Fallstudie bereits erledigt – das Ergebnis liegt in `regions.csv`.

### Die aufbereiteten Datentabellen

**`regions.csv` – Kundenregionen:**

| Spalte | Bedeutung |
|--------|-----------|
| `regionID` | eindeutige ID der Region |
| `zipCode` | PLZ des Regionszentrums |
| `city` | Stadt im Regionszentrum |
| `lat`, `lon` | Koordinaten des Regionszentrums |
| `yearlyDemand` | jährlich benötigte Lieferungen/Stops in der Region |
| `minutesPerStop` | durchschnittliche Zeit pro Kunde (inkl. Fahrt *zwischen* Kunden und Zeit *beim* Kunden) |

**`warehouses.csv` – Cash Center:**

| Spalte | Bedeutung |
|--------|-----------|
| `warehouseID` | eindeutige ID des Centers |
| `city` | Standort |
| `fixedCosts` | jährliche Fixkosten, falls betrieben |
| `lat`, `lon` | Koordinaten |

(Beispielwerte zeigen: Barcelona ist mit 15.012.000 € Fixkosten deutlich teurer als kleinere
Center wie Vitoria mit 1.344.000 € – Fixkosten variieren stark nach Standortgröße.)

**`shifts.csv` – Verbindungen Center ↔ Region (entspricht der $c_{ij}$-Matrix):**

| Spalte | Bedeutung |
|--------|-----------|
| `warehouseID` | ID des Centers $i$ |
| `regionID` | ID der Region $j$ |
| `transportationCosts` | **jährliche Kosten**, um die Nachfrage der Region zu bedienen – *zunächst nur Platzhalter (= travelTime), muss von dir korrekt befüllt werden!* |
| `travelTime` | Fahrzeit Center → Region in Minuten (einfache Strecke) |

Die Tabelle enthält **alle Kombinationen** Center × Region.

---

## 6. ANALYSIS, Problem 1 – Transportkosten $c_{ij}$ schätzen

Das ist die **erste echte Analyseaufgabe**: Wie wird aus Fahrzeit, Nachfrage und Produktivität
einer Region eine sinnvolle jährliche Transportkostenzahl?

### Gegebene Annahmen

- Eine **Schicht** dauert **450 Minuten** (8 h − 30 min für Be-/Entladen).
- Eine Schicht = **ein LKW + Crew von drei**; **Kosten je Schicht = 480 €** (Löhne, Treibstoff,
  Abschreibung, Wartung).

### Die Herleitung Schritt für Schritt

**Schritt 1 – Wie viel Zeit bleibt in einer Schicht für die Kundenbedienung?**
Der LKW fährt zur Region (`travelTime`) und wieder zurück (`travelTime`), also **Hin- und
Rückfahrt** $= 2 \cdot \text{travelTime}$. Die restliche Zeit steht für Stops zur Verfügung:

$$\text{verfügbare Bedienzeit pro Schicht} = 450 - 2 \cdot \text{travelTime}_{ij}$$

**Schritt 2 – Wie viele Stops schafft eine Schicht?**
Jeder Stop kostet `minutesPerStop` (inkl. Fahrt zwischen den Kunden):

$$\text{Stops pro Schicht}_{ij} = \frac{450 - 2 \cdot \text{travelTime}_{ij}}{\text{minutesPerStop}_j}$$

**Schritt 3 – Wie viele Schichten braucht eine Region pro Jahr?**
Die Region benötigt `yearlyDemand` Stops im Jahr:

$$\text{Schichten pro Jahr}_{ij} = \frac{\text{yearlyDemand}_j}{\text{Stops pro Schicht}_{ij}}$$

**Schritt 4 – Jährliche Transportkosten:**
Jede Schicht kostet 480 €:

$$\boxed{\;c_{ij} = 480 \cdot \frac{\text{yearlyDemand}_j \cdot \text{minutesPerStop}_j}{\;450 - 2 \cdot \text{travelTime}_{ij}\;}\;}$$

### Interpretation & wichtige Feinheiten

- **Je weiter die Region** (großes `travelTime`), desto kleiner der Nenner → desto **teurer**
  die Bedienung. Ökonomisch genau richtig.
- **Produktivität** der Region steckt in `minutesPerStop`: Regionen, in denen Kunden dicht
  beieinander liegen (kleines `minutesPerStop`), sind günstiger zu bedienen.
- **Infeasibilität:** Ist $2 \cdot \text{travelTime}_{ij} \ge 450$, wird der Nenner null oder
  negativ → die Region kann **nicht** in einer einzelnen Schicht von diesem Center bedient
  werden. Solche Verbindungen müssen ausgeschlossen werden (sehr hohe Kosten / „unendlich"
  setzen bzw. die Zuordnung verbieten).
- **Rundung:** Streng genommen sind „Stops pro Schicht" und „Schichten pro Jahr" ganzzahlig
  (man kann keine halbe Schicht fahren). Für die strategische Kostenschätzung wird meist mit
  kontinuierlichen Werten gerechnet; man kann optional aufrunden (`ceil`), um realistischer zu
  sein. Das ist eine bewusste **Approximation**.

> In `data/shifts_with_costs.csv` liegt eine Referenzversion mit bereits geschätzten Kosten –
> ideal, um die eigene Berechnung zu **validieren** (Benchmark).

### Pseudocode

```python
import pandas as pd

shifts    = pd.read_csv("data/shifts.csv")
regions   = pd.read_csv("data/regions.csv").set_index("regionID")

SHIFT_MIN  = 450     # nutzbare Minuten pro Schicht
SHIFT_COST = 480     # Euro pro Schicht

def transport_cost(row):
    region = regions.loc[row["regionID"]]
    usable = SHIFT_MIN - 2 * row["travelTime"]      # Zeit für Stops
    if usable <= 0:
        return float("inf")                          # Region nicht erreichbar
    stops_per_shift = usable / region["minutesPerStop"]
    shifts_per_year = region["yearlyDemand"] / stops_per_shift
    return shifts_per_year * SHIFT_COST

shifts["transportationCosts"] = shifts.apply(transport_cost, axis=1)
shifts.to_csv("data/shifts_my_costs.csv", index=False)
```

---

## 7. ANALYSIS, Problem 2 – Eine erste Lösung

Mit den berechneten $c_{ij}$ wird das bereitgestellte Optimierungs-Notebook ausgeführt:

1. **Modelllogik verstehen** (Warehouse-Location-Struktur, jetzt mit Regionen statt Einzelkunden).
2. **Modell rechnen lassen** mit den eigenen Transportkosten.
3. **Ergebnisse interpretieren** und mit der Karten-Visualisierung **plausibilisieren**:
   Liegen die offen gehaltenen Center sinnvoll verteilt? Werden Regionen jeweils dem
   *nächstgelegenen* sinnvollen Center zugeordnet? Gibt es Center, die kaum Volumen tragen?

Diese Plausibilitätsprüfung ist Teil guter Praxis: Ein Optimierungsergebnis, das man nicht
geografisch/ökonomisch nachvollziehen kann, ist verdächtig (oft ein Daten- oder Modellfehler).

---

## 8. ANALYSIS, Problem 3 – Kapazitäten & nicht-konstante Fixkosten

### Das Problem mit dem Standardmodell

Das klassische Warehouse-Location-Modell unterstellt:
- jede Anlage hat eine **gegebene, feste Kapazität**,
- die Fixkosten dafür sind eine **Konstante** $f_i$.

Bei CashLog stimmt das nicht: Da es eine **strategische, langfristige** Entscheidung ist, kann
Kapazität durch Investitionen **vergrößert oder verkleinert** werden – Kapazität ist (zumindest
implizit) eine **Entscheidungsvariable**. Damit können die jährlichen Kosten nicht unabhängig vom
Volumen sein.

### Realistische Kostenstruktur

Sinnvoll ist die Annahme:
- einen **jährlichen Fixkostenanteil** je Center (Overhead, Miete, Versicherung), der zwar von
  der Kapazität abhängt, aber **nicht proportional** mit dem Volumen wächst → **Skaleneffekte**;
- einen **volumenabhängigen Kostenanteil** (z. B. Personal zur Verarbeitung), der ebenfalls
  nicht konstant ist: Ein sehr großes Center hat **niedrigere Kosten pro Lieferung** als ein
  kleines → ebenfalls Skaleneffekte.

Insgesamt entsteht eine **stückweise lineare** Gesamtkostenkurve über dem Volumen.

### Modellierungstrick: diskrete Center-Typen

Man definiert **fünf Center-Typen** nach Volumen:
**very small (v), small (s), medium (m), large (l), huge (h)**.

- Index $t \in \{v,s,m,l,h\}$ für den Typ.
- Jeder Typ hat eine **Volumen-Untergrenze** $V_t^{lb}$ und **-Obergrenze** $V_t^{ub}$
  (Beispiel: $V_v^{lb}=0,\ V_v^{ub}=20000$; $V_s^{lb}=20001,\ V_s^{ub}=45000$; …).
- Jeder Typ hat **fixe Jahreskosten** $c_t^{fix}$ und **variable Kosten pro Lieferung** $c_t^{var}$.

So approximiert man die geschwungene Skaleneffekt-Kurve durch mehrere lineare Stücke (eines je
Typ) – daher **piecewise linear costs**. Größere Typen haben höhere Fixkosten, aber niedrigere
variable Kosten pro Lieferung.

### Neue Variablen

- $y_{it}$ = binär: ist Center $i$ vom Typ $t$? (Ist das Center geschlossen, ist $y_{it}=0$ für
  alle $t$.)
- $z_{it}$ = Anzahl Lieferungen, die Center $i$ als Typ $t$ abwickelt.

Die Jahreskosten eines Centers schreibt man dann als:

$$\sum_i \sum_t c_t^{fix}\, y_{it} \;+\; \sum_i \sum_t c_t^{var}\, z_{it}$$

---

## 9. Das erweiterte CashLog-Modell (mit stückweise linearen Kosten)

### Gegenüberstellung

**Basismodell:**

$$\min \; C(X,Y) = \sum_i\sum_j x_{ij} c_{ij} + \sum_i f_i y_i$$

$$x_{ij} \le y_i \quad \forall i,j \qquad \sum_i x_{ij} = 1 \quad \forall j$$
$$x_{ij}\in\{0,1\},\quad y_i\in\{0,1\}$$

**Erweitertes Modell:**

$$\min \; C(X,Y,Z) = \sum_i\sum_j x_{ij} c_{ij} \;+\; \sum_i\sum_t c_t^{fix} y_{it} \;+\; \sum_i\sum_t c_t^{var} z_{it}$$

| Nebenbedingung | Bedeutung |
|----------------|-----------|
| $x_{ij} \le \sum_t y_{it} \quad \forall i,j$ | Region $j$ kann $i$ nur zugeordnet werden, wenn $i$ (in *irgendeinem* Typ) offen ist. |
| $\sum_i x_{ij} = 1 \quad \forall j$ | Jede Region wird genau einem Center zugeordnet. |
| $\sum_t z_{it} = \sum_j x_{ij} d_j \quad \forall i$ | Das von $i$ verarbeitete Volumen entspricht der zugeordneten Nachfrage. |
| $z_{it} \ge V_t^{lb}\, y_{it} \quad \forall i,t$ | Wenn $i$ Typ $t$ ist, muss das Volumen die Untergrenze des Typs erreichen. |
| $z_{it} \le V_t^{ub}\, y_{it} \quad \forall i,t$ | …und die Obergrenze nicht überschreiten. |
| $x_{ij}\in\{0,1\},\ y_{it}\in\{0,1\}$ | Binärvariablen. |

**Wichtige Ergänzung (auf den Folien angekündigt):** Eine Nebenbedingung
$\sum_t y_{it} \le 1 \;\forall i$ stellt sicher, dass jedes Center **höchstens einen Typ**
annimmt. Andernfalls könnte das Modell ein Center künstlich „aufteilen".

### Warum das so funktioniert

- Wählt das Modell für Center $i$ den Typ $t$ (also $y_{it}=1$), erzwingen die beiden
  $z_{it}$-Schranken, dass das verarbeitete Volumen genau in das **Volumenfenster** dieses Typs
  passt.
- Über $\sum_t z_{it} = \sum_j x_{ij} d_j$ ist das verarbeitete Volumen an die tatsächlich
  zugeordnete Nachfrage gekoppelt.
- Die Zielfunktion wählt automatisch **Standort, Typ (= implizite Kapazität) und Zuordnung**
  gleichzeitig kostenminimal – inklusive Skaleneffekten.

Vom Modelltyp her ist das ein **gemischt-ganzzahliges lineares Programm (MILP)**.

---

## 10. CONCLUSION & Aufgabe für Case Study 2

Das eigentliche Ziel ist nicht *eine* Lösung, sondern eine **robuste Empfehlung**. Die
abschließende Aufgabe besteht aus zwei Teilen.

### Teil 1 – Transportkosten schätzen
Die in Abschnitt 6 hergeleitete Methode umsetzen und gegen `data/shifts_with_costs.csv`
validieren.

### Teil 2 – Umfassende Sensitivitätsanalyse für robuste Empfehlungen

Die Schlüsselfrage: *Wie verändert sich die optimale Netzstruktur, wenn sich zentrale Annahmen
ändern?* Drei Stoßrichtungen sind vorgegeben:

1. **Unsicherheit bei der künftigen Bargeldnutzung / Nachfrage.**
   - Bargeld wird tendenziell weniger genutzt. Was passiert bei −10 %, −30 %, −50 % Nachfrage?
   - Ansatz: das Modell für mehrere **Nachfrageszenarien** rechnen und prüfen, welche Center in
     *allen* Szenarien offen bleiben (= robuste „Behalten"-Kandidaten) und welche nur in
     einzelnen Szenarien.

2. **Steigende Löhne und Treibstoffkosten.**
   - Diese erhöhen die **Schichtkosten** (die 480 €) und damit alle $c_{ij}$.
   - Höhere Transportkosten machen **kurze Wege wertvoller** → tendenziell spricht das für *mehr*
     offene, dezentrale Center. Sensitivität der Lösung gegenüber dem 480-€-Parameter prüfen.

3. **Neue Technologien** (z. B. autonome Drohnenlieferungen, selbstfahrende LKW).
   - Diese würden die Kostenstruktur grundlegend verändern (z. B. wegfallende Crew-Kosten,
     andere Reichweiten/Schichtlängen).
   - Was bedeutet das für die strategische Entscheidung? Würde man weniger, dafür größere Center
     bevorzugen?

**Idee von „Robustheit":** Eine Empfehlung ist gut, wenn sie nicht nur für *eine* Punktschätzung
optimal ist, sondern über ein **plausibles Spektrum** an Annahmen hinweg gut (oder zumindest
nicht schlecht) bleibt. Praktisch identifiziert man:
- Center, die **über alle Szenarien** offen bleiben → klar behalten.
- Center, die **nie** offen sind → klar schließen.
- Center, deren Status **vom Szenario abhängt** → die eigentlich kritischen Entscheidungen, die
  zusätzliche Überlegung/Daten erfordern.

---

## 11. Der rote Faden in einem Satz

> CashLog will von 42 teuren Cash Centern auf die kostenoptimale Teilmenge schrumpfen; dazu
> übernimmt man die Logik des Warehouse-Location-Modells, macht es praxistauglich durch
> **Clustering** der 42.000 Kunden zu Regionen, eine **schichtbasierte Schätzung der
> Transportkosten** ($c_{ij} = 480 \cdot \text{Demand} \cdot \text{minPerStop} / (450 - 2\cdot
> \text{travelTime})$) und eine **stückweise lineare Kostenstruktur** für variable Kapazitäten –
> und sichert die Empfehlung über eine **Sensitivitätsanalyse** gegen Unsicherheit ab.

---

## Anhang: Symbolübersicht

| Symbol | Bedeutung |
|--------|-----------|
| $i$ | Cash Center / Warehouse |
| $j$ | Kundenregion |
| $t$ | Center-Typ $\in \{v,s,m,l,h\}$ |
| $x_{ij}$ | Zuordnung Region $j$ → Center $i$ (binär im CashLog-Modell) |
| $y_i$ / $y_{it}$ | Center $i$ offen / Center $i$ ist vom Typ $t$ |
| $z_{it}$ | von Center $i$ als Typ $t$ verarbeitetes Volumen |
| $c_{ij}$ | jährliche Transportkosten Center $i$ → Region $j$ |
| $d_j$ / yearlyDemand | jährliche Nachfrage (Stops) der Region $j$ |
| $f_i$ | konstante Fixkosten (Basismodell) |
| $c_t^{fix}$, $c_t^{var}$ | Fix-/variable Kosten je Center-Typ $t$ |
| $V_t^{lb}$, $V_t^{ub}$ | Volumen-Unter-/Obergrenze des Typs $t$ |
| $cap_i$ | Kapazität (Basismodell) |
| 450 | nutzbare Minuten pro Schicht (8 h − 30 min) |
| 480 € | Kosten pro Schicht (LKW + 3er-Crew) |
