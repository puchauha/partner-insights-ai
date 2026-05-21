
# Partner Insights AI

AI-driven operational analytics assistant that converts vendor outage emails into decision-ready intelligence.

## Overview

Partner Insights AI demonstrates how AI and prompt-driven analytics can transform unstructured vendor outage communication into structured operational insights.

The solution enables conversational analytics, dynamic visualization generation, and partner-level operational intelligence without predefined dashboards.

---

## Key Features

- AI-assisted parsing of outage-related emails
- Structured operational attribute extraction
- Conversational analytics over outage history
- Dynamic pandas and chart generation
- Heatmaps, scatter plots, pie charts, and tables
- Streamlit-based operational analytics UI

---

## Technology Stack

- Streamlit
- LangGraph
- GPT-4o mini
- Pandas
- Matplotlib
- Seaborn

---

## High-Level Flow

Vendor Outage Emails
    ↓
AI Parsing & Extraction
    ↓
Structured Operational Dataset
    ↓
Prompt-Driven Analytics Engine
    ↓
Dynamic Visualization Layer
    ↓
Streamlit UI

---

## Future Enhancements

- Real email ingestion pipeline
- AWS deployment
- Observability and tracing
- Agent accuracy evaluation
- Token optimization
- Governance and guardrails
- Memory-aware analytics

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run partner_insights_ui.py
```

---

## Example Queries

- Partner Outage details in a table
- heatmap of  issue details for each partner
- outage in stacked bar graph
- outage in stacked bar graph in red and green
- partner outage pie chart in grey scale
- outage count and outage duration for oceanic
- average outage duration for all partners desc

---

## Author

Purusharth Chauhan
