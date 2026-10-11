# Zomato Restaurant Recommendation System (Content-Based, NLP)

Pick a restaurant - or just describe what you want - and get restaurants with **similar reviews**, **sorted from the highest rating**.

## Features of the live app
| Tab | What it does |
|---|---|
| Find similar | Search a restaurant, get similar ones. Cards, **map** and table / chart views; **"Why this?"** shows the shared review phrases; 👍 / 👎 feedback |
| Describe what you want | Free-text search, e.g. *"romantic rooftop with live music"* |
| Analytics | Market dashboard: best-rated areas, cuisines, price vs rating, rating distribution, votes vs rating |
| Feedback | Live summary of the 👍 / 👎 votes saved in `feedback.db` (SQLite) + CSV download |

Sidebar filters (all tabs): number of results, similarity pool, minimum rating, max cost, locations, cuisines, exclude same-chain branches.

## Files
| File | Purpose |
|---|---|
| `zomato_recommender.ipynb` | Cleaning, EDA, NLP, TF-IDF, similarity, evaluation; writes `artifacts/` |
| `app.py` | Streamlit app |
| `text_utils.py` | Text cleaning shared by notebook and app (so user input is processed exactly like the training data) |
| `geo.py` | Approximate neighbourhood coordinates for the map (dataset has none) |
| `artifacts/` | Created by the notebook: restaurant table, neighbour index, TF-IDF vectorizer + matrix, word map |
| `zomato.csv` | Dataset (not included - put it next to the notebook) |

## Run locally
```bash
pip install -r requirements.txt
jupyter notebook zomato_recommender.ipynb      # Run All -> creates artifacts/  (~1 min)
streamlit run app.py                           # opens http://localhost:8501
```
If the app says the model files cannot be loaded, re-run the notebook: the saved vectorizer is tied to your scikit-learn version.

## Put it online (client demo link)
**Streamlit Community Cloud (free):** push this folder (including `artifacts/`, about 29 MB) to a GitHub repo, then on share.streamlit.io choose the repo and `app.py` as the main file. The app gets a public URL. Note: files written at runtime (`feedback.db`) are not permanent there; for production use a hosted database.

**Docker:**
```bash
docker build -t zomato-recommender .
docker run -p 8501:8501 zomato-recommender
```
