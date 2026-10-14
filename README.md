# 📊 Multi-Channel Marketing Analytics & Sales Prediction Engine

An end-to-end Machine Learning and Business Intelligence web application built to analyze advertising efficiency, forecast retail sales, and provide actionable, data-driven marketing recommendations.

---

## 🚀 Key Features

1. **Exploratory Data Analysis (EDA):** Interactive data inspection, statistical summaries, correlation heatmaps, and dynamic channel-vs-sales scatter plots.
2. **Live Prediction Simulator:** A real-time inference engine that allows stakeholders to adjust custom campaign budgets (TV, Radio, Newspaper) and instantly compute projected sales volume and budget distributions.
3. **Business ROI & Strategic Insights:** Evaluation metrics ($R^2$ score and Mean Absolute Error), actual vs. predicted regression fits, and executive recommendations on channel efficiency and budget reallocation.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-Learn (Linear Regression), NumPy, Pandas
* **Data Visualization:** Matplotlib, Seaborn
* **Frontend / UI:** Streamlit (Custom Mint & Aqua Theme)
* **Architecture:** Modular separation of backend data pipelines from the frontend user interface

---

## 📂 Project Architecture

```text
Sales_Prediction/
│
├── Advertising.csv             # Retail multi-channel advertising dataset
├── .streamlit/
│   └── config.toml             # Custom mint & aqua UI color palette configuration
├── backend_ and_MODEL/
│   └── model_backend.py        # ML data pipeline, training logic, and inference engine
└── FrontEnd/
    └── app.py                  # Main tabbed dashboard interface
