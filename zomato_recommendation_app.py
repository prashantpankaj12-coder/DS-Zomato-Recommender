"""
Zomato Restaurant Recommender - Streamlit app (live-demo version)
------------------------------------------------------------------
Run the notebook `zomato_recommender.ipynb` once (it creates ./artifacts), then:

    streamlit run app.py

Tabs
  * Find similar          - pick a restaurant, get restaurants with similar reviews, best rated first
  * Describe what you want - free-text search, e.g. "romantic rooftop with live music"
  * Analytics             - market dashboard (ratings, cuisines, prices, areas)
  * Feedback              - thumbs up / down are stored in a local SQLite file and summarised here
Every result list can be viewed as cards, on a map, or as a table / chart, and each card has a
"Why this?" pop-over that shows the review phrases behind the match.
"""
import html
import re
import sqlite3
import zlib
from urllib.parse import quote_plus
from contextlib import closing
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import scipy.sparse as sp
import streamlit as st

from text_utils import clean_text, readable

ART = Path(__file__).parent / "artifacts"
DB_PATH = Path(__file__).parent / "feedback.db"
RED, NAVY, GREEN = "#E23744", "#1F3A6B", "#267E3E"

st.set_page_config(page_title="Zomato Recommender", page_icon="🍽️", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      .hero {background: linear-gradient(90deg,#E23744,#b81d2c); color:#fff; padding:1.3rem 1.6rem;
             border-radius:14px; margin-bottom:1rem;}
      .hero h1 {margin:0; font-size:1.8rem; color:#fff;}
      .hero p {margin:.3rem 0 0 0; opacity:.92;}
      .card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:.85rem 1.1rem; margin-top:.7rem;}
      .card h4 {margin:0 0 .2rem 0;}
      .pill {display:inline-block; padding:.05rem .55rem; border-radius:999px; font-size:.78rem;
             background:rgba(226,55,68,.12); color:#E23744; margin:.2rem .3rem 0 0;}
      .pill.why {background:rgba(31,58,107,.12); color:#1F3A6B;}
      .rate {float:right; background:#267E3E; color:#fff; padding:.15rem .6rem; border-radius:8px; font-weight:700;}
      .muted {opacity:.7; font-size:.85rem;}
      .pill.ok {background:rgba(38,126,62,.14); color:#267E3E;}
      .pill.off {background:rgba(128,128,128,.14); color:#777;}
      .act {display:inline-block; margin:.5rem .5rem 0 0; padding:.22rem .75rem; border:1px solid rgba(128,128,128,.4);
            border-radius:8px; font-size:.82rem; text-decoration:none !important; color:inherit !important;}
      .act:hover {border-color:#E23744; color:#E23744 !important;}
      .quote {opacity:.8; font-style:italic; font-size:.88rem; margin-top:.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- data
REQUIRED = ["restaurants.pkl", "neighbors.npz", "tfidf.joblib", "tfidf_matrix.npz", "unstem.json"]
if not all((ART / f).exists() for f in REQUIRED):
    st.error("Model files not found. Run **zomato_recommender.ipynb** first - it creates the `artifacts/` folder.")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def load():
    import json
    df = pd.read_pickle(ART / "restaurants.pkl")
    nb = np.load(ART / "neighbors.npz")
    tfidf = joblib.load(ART / "tfidf.joblib")
    X = sp.load_npz(ART / "tfidf_matrix.npz").tocsr()
    unstem = json.load(open(ART / "unstem.json"))
    return df, nb["idx"], nb["sim"], tfidf, X, np.array(tfidf.get_feature_names_out()), unstem


try:
    df, NB_IDX, NB_SIM, TFIDF, X, TERMS, UNSTEM = load()
except Exception as exc:        # most often a scikit-learn version mismatch with the saved vectorizer
    st.error("Could not load the saved model files. Re-run **zomato_recommender.ipynb** (Run All) on this machine "
             f"so they are rebuilt with your library versions.\n\nDetails: {exc}")
    st.stop()
VOCAB = TFIDF.vocabulary_
LABEL_TO_IDX = {l: i for i, l in enumerate(df["label"])}
COST_MAX = int(df["cost"].max())


# ----------------------------------------------------------------------------- feedback store (SQLite)
def _db():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS feedback (ts TEXT, mode TEXT, query TEXT, restaurant TEXT, vote INTEGER)")
    return con


def record_vote(key, mode, query, restaurant, vote):
    with closing(_db()) as con:
        con.execute("INSERT INTO feedback VALUES (?,?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), mode, query, restaurant, vote))
        con.commit()
    st.session_state.setdefault("voted", {})[key] = vote
    st.toast("Thanks for the feedback! 👍" if vote > 0 else "Thanks - we'll use this to improve 👎")


def load_feedback():
    with closing(_db()) as con:
        return pd.read_sql("SELECT * FROM feedback ORDER BY ts DESC", con)


# ----------------------------------------------------------------------------- recommender
def filter_mask(min_rating, max_cost, locations, cuisines, only_online=False, only_book=False):
    m = (df["rate"] >= min_rating).to_numpy().copy()
    if only_online:
        m &= (df["online_order"] == "Yes").to_numpy()
    if only_book:
        m &= (df["book_table"] == "Yes").to_numpy()
    if max_cost < COST_MAX:
        m &= (df["cost"].isna() | (df["cost"] <= max_cost)).to_numpy()
    if locations:
        m &= df["location"].isin(locations).to_numpy()
    if cuisines:
        pat = "|".join(map(re.escape, cuisines))
        m &= df["cuisines"].str.contains(pat, case=False).to_numpy()
    return m


def recommend_similar(i, top_n, pool, mask, exclude_brand):
    idx, sims = NB_IDX[i], NB_SIM[i]
    keep = mask[idx]
    if exclude_brand:
        keep &= (df["name"].to_numpy()[idx] != df.at[i, "name"])
    idx, sims = idx[keep][:pool], sims[keep][:pool]
    out = df.iloc[idx].copy()
    out["similarity"] = sims
    return out.sort_values(["rate", "votes"], ascending=False).head(top_n).reset_index(drop=True)


def recommend_text(qvec, top_n, pool, mask):
    sims = (X @ qvec.T).toarray().ravel()
    sims = np.where(mask, sims, -1.0)
    top = np.argsort(-sims)[:pool]
    top = top[sims[top] > 0]
    out = df.iloc[top].copy()
    out["similarity"] = sims[top]
    return out.sort_values(["rate", "votes"], ascending=False).head(top_n).reset_index(drop=True)


def why_terms(qrow, j, k=5):
    """Review words / phrases that the query and restaurant j have most in common (readable form)."""
    prod = qrow.multiply(X[j]).tocsr()
    if prod.nnz == 0:
        return []
    chosen = []
    for o in np.argsort(-prod.data)[:14]:
        term = readable(TERMS[prod.indices[o]], UNSTEM)
        if any(term in c or c in term for c in chosen):      # skip "live" when "live music" is already shown
            continue
        chosen.append(term)
        if len(chosen) == k:
            break
    return chosen


# ----------------------------------------------------------------------------- UI pieces
def _actions(row):
    """Zomato page / directions / map / call links for a restaurant."""
    e = html.escape
    where = f"{row['name']}, {row['address']}" if row["address"] else f"{row['name']}, {row['location']}, Bangalore"
    dest = quote_plus(where)
    links = []
    if row["url"]:
        links.append(f"<a class='act' href='{e(row['url'])}' target='_blank' rel='noopener'>🔗 Zomato page</a>")
    links.append(f"<a class='act' href='https://www.google.com/maps/dir/?api=1&amp;destination={dest}' target='_blank' rel='noopener'>🧭 Directions</a>")
    links.append(f"<a class='act' href='https://www.google.com/maps/search/?api=1&amp;query={dest}' target='_blank' rel='noopener'>🗺️ View on map</a>")
    first = row["phone"].split(",")[0].strip() if row["phone"] else ""
    if first:
        links.append(f"<a class='act' href='tel:{re.sub(r'[^0-9+]', '', first)}'>📞 {e(first)}</a>")
    return "".join(links)


def card(row, rank=None, similarity=True):
    e = html.escape
    title = f"{rank}. {e(row['name'])}" if rank else e(row["name"])
    cost = f"₹{int(row['cost'])} for two" if pd.notna(row["cost"]) else "cost n/a"
    sim = f" &nbsp;·&nbsp; {row['similarity']*100:.0f}% review match" if similarity and "similarity" in row else ""
    pills = "".join(f"<span class='pill'>{e(c.strip())}</span>" for c in str(row["cuisines"]).split(",")[:5])
    online = "<span class='pill ok'>🛵 Online ordering</span>" if row["online_order"] == "Yes" else "<span class='pill off'>No online ordering</span>"
    book = "<span class='pill ok'>📅 Table booking</span>" if row["book_table"] == "Yes" else "<span class='pill off'>No table booking</span>"
    dishes = f"<div class='muted'>Popular: {e(row['dish_liked'][:110])}</div>" if row["dish_liked"] else ""
    address = f"<div class='muted'>🏠 {e(row['address'])}</div>" if row["address"] else ""
    st.markdown(
        f"""<div class='card'><span class='rate'>★ {row['rate']:.1f}</span><h4>{title}</h4>
        <div class='muted'>📍 {e(row['location'])} &nbsp;·&nbsp; {e(row['rest_type'])} &nbsp;·&nbsp; {cost}
        &nbsp;·&nbsp; {int(row['votes']):,} votes{sim}</div>{address}
        <div>{pills}{online}{book}</div>{dishes}
        <div class='quote'>“{e(row['top_review'])}”</div><div>{_actions(row)}</div></div>""",
        unsafe_allow_html=True,
    )


def _jitter(label):
    c = zlib.crc32(label.encode())
    return ((c % 1000) / 1000 - .5) * 0.010, (((c // 1000) % 1000) / 1000 - .5) * 0.010


def map_view(recs, selected=None, mode="x"):
    pts = recs.assign(kind="Recommended", color=[[31, 58, 107, 220]] * len(recs))
    if selected is not None:
        s = selected.to_frame().T.assign(kind="Selected", color=[[226, 55, 68, 255]])
        pts = pd.concat([pts, s], ignore_index=True)
    pts = pts.dropna(subset=["lat", "lon"]).copy()
    if pts.empty:
        st.info("No map position available for these restaurants.")
        return
    jit = pts["label"].map(_jitter)
    pts["lat"] = pts["lat"].astype(float) + jit.map(lambda t: t[0])
    pts["lon"] = pts["lon"].astype(float) + jit.map(lambda t: t[1])
    pts["rate_txt"] = pts["rate"].map("{:.1f}".format)
    layer = pdk.Layer("ScatterplotLayer", pts[["name", "location", "rate_txt", "kind", "lat", "lon", "color"]],
                      get_position="[lon, lat]", get_fill_color="color", get_radius=180,
                      radius_min_pixels=8, radius_max_pixels=16, pickable=True, stroked=True,
                      get_line_color=[255, 255, 255], line_width_min_pixels=2)
    view = pdk.ViewState(latitude=float(pts["lat"].mean()), longitude=float(pts["lon"].mean()), zoom=10.6)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                             tooltip={"html": "<b>{name}</b><br/>{location} · ★ {rate_txt}<br/><i>{kind}</i>"}),
                    height=480, key=f"map_{mode}")
    st.caption("🔴 selected restaurant · 🔵 recommendations. Pins mark the neighbourhood (approximate) - use 🧭 Directions for the exact address.")


def results(recs, mode, query_label, qrow, selected=None):
    """Cards / map / table views of a recommendation list, with 'why' pop-overs and feedback buttons."""
    t_cards, t_map, t_table = st.tabs(["🃏 Cards", "🗺️ Map", "📋 Table & chart"])
    with t_cards:
        for r, (_, row) in enumerate(recs.iterrows(), 1):
            card(row, r)
            c_up, c_down, c_why, _ = st.columns([1, 1, 2.4, 8])
            key = f"{mode}|{query_label}|{row['label']}"
            done = st.session_state.get("voted", {}).get(key)
            c_up.button("👍" if done != 1 else "👍 ✓", key=f"up|{key}", disabled=done is not None,
                        on_click=record_vote, args=(key, mode, query_label, row["label"], 1))
            c_down.button("👎" if done != -1 else "👎 ✓", key=f"dn|{key}", disabled=done is not None,
                          on_click=record_vote, args=(key, mode, query_label, row["label"], -1))
            with c_why.popover("Why this?"):
                terms = why_terms(qrow, LABEL_TO_IDX[row["label"]])
                if terms:
                    st.markdown("Reviews of both places talk about:")
                    st.markdown("".join(f"<span class='pill why'>{html.escape(t)}</span>" for t in terms),
                                unsafe_allow_html=True)
                else:
                    st.write("No strong shared phrases.")
                st.caption(f"Review similarity: {row['similarity']*100:.0f}%  ·  rating ★ {row['rate']:.1f} "
                           f"from {int(row['votes']):,} votes")
    with t_map:
        map_view(recs, selected, mode)
        st.markdown("**Ready to go?** Get directions from your current location:")
        for r, (_, row) in enumerate(recs.iterrows(), 1):
            dest = quote_plus(f"{row['name']}, {row['address']}" if row["address"] else f"{row['name']}, {row['location']}, Bangalore")
            st.markdown(f"{r}. **{row['name']}** ({row['location']}) · "
                        f"[🧭 Directions](https://www.google.com/maps/dir/?api=1&destination={dest}) · "
                        f"[🗺️ View on map](https://www.google.com/maps/search/?api=1&query={dest})")
    with t_table:
        left, right = st.columns([3, 2], gap="large")
        show = recs[["name", "location", "cuisines", "rate", "votes", "cost", "similarity"]].copy()
        show["similarity"] = (show["similarity"] * 100).round(0).astype(int).astype(str) + "%"
        with left:
            st.dataframe(show, hide_index=True, width="stretch")
            st.download_button("Download as CSV", show.to_csv(index=False).encode(), "recommendations.csv", "text/csv", key=f"dl_{mode}")
        with right:
            chart = recs.assign(restaurant=recs["name"] + " (" + recs["location"] + ")")
            fig = px.bar(chart.iloc[::-1], x="rate", y="restaurant", orientation="h", range_x=[0, 5],
                         color_discrete_sequence=[RED], labels={"rate": "Rating", "restaurant": ""})
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch", key=f"bar_{mode}")


# ----------------------------------------------------------------------------- sidebar filters
with st.sidebar:
    st.header("Filters")
    top_n = st.slider("Number of recommendations", 3, 20, 10)
    pool = st.slider("Similarity pool", 10, 300, 50, step=10,
                     help="How many of the most review-similar restaurants to consider before sorting by rating. "
                          "Smaller = more similar, larger = higher rated but less similar.")
    min_rating = st.slider("Minimum rating", 0.0, 5.0, 0.0, step=0.1)
    max_cost = st.slider("Max cost for two (₹)", 100, COST_MAX, COST_MAX, step=100)
    locations = st.multiselect("Only these locations", sorted(df["location"].unique()))
    all_cuisines = sorted({c.strip() for s in df["cuisines"] for c in s.split(",") if c.strip() != "Unknown"})
    cuisines = st.multiselect("Only these cuisines", all_cuisines)
    only_online = st.checkbox("🛵 Online ordering available")
    only_book = st.checkbox("📅 Table booking available")
    exclude_brand = st.checkbox("Exclude other branches of the same restaurant", value=True)

MASK = filter_mask(min_rating, max_cost, locations, cuisines, only_online, only_book)

st.markdown(
    "<div class='hero'><h1>🍽️ Zomato Restaurant Recommender</h1>"
    "<p>Find restaurants by what people say in their reviews - pick a place you love, or just describe what you want.</p></div>",
    unsafe_allow_html=True,
)
tab_sim, tab_text, tab_an, tab_fb, tab_about = st.tabs(
    ["🔎 Find similar", "✍️ Describe what you want", "📊 Analytics", "💬 Feedback", "ℹ️ About"])

# ----------------------------------------------------------------------------- tab 1: find similar
with tab_sim:
    st.caption("Quick demo:")
    ex_cols = st.columns(5)
    for col, name in zip(ex_cols, ["Truffles", "Toit", "Jalsa", "Onesta", "Empire Restaurant"]):
        col.button(name, key=f"ex_{name}", width="stretch",
                   on_click=lambda n=name: st.session_state.update(search_q=n))
    search = st.text_input("Search restaurant name", key="search_q", placeholder="e.g. Truffles, Jalsa, Toit ...")
    if search.strip():
        options = df.loc[df["name"].str.contains(search.strip(), case=False, regex=False), "label"].tolist()
    else:
        options = df.sort_values("votes", ascending=False)["label"].head(200).tolist()
    if not options:
        st.warning("No restaurant matches that name. Try a shorter search.")
    else:
        choice = st.selectbox(f"Select restaurant ({len(options)} match{'es' if len(options) != 1 else ''})", options)
        i = LABEL_TO_IDX[choice]
        sel = df.iloc[i]
        st.subheader("You selected")
        card(sel, similarity=False)
        recs = recommend_similar(i, top_n, pool, MASK, exclude_brand)
        st.subheader(f"Recommended for you ({len(recs)})")
        if recs.empty:
            st.info("No restaurants match these filters. Relax them or increase the similarity pool.")
        else:
            results(recs, "similar", choice, X[i], selected=sel)

# ----------------------------------------------------------------------------- tab 2: free text
with tab_text:
    st.caption("Try one:")
    examples = ["romantic rooftop with live music", "cheap and tasty biryani", "good place for kids birthday party",
                "craft beer and burgers with friends"]
    ex_cols = st.columns(len(examples))
    for col, ex in zip(ex_cols, examples):
        col.button(ex, key=f"tx_{ex}", width="stretch", on_click=lambda e=ex: st.session_state.update(desc_q=e))
    query = st.text_input("Describe what you are looking for", key="desc_q",
                          placeholder="e.g. quiet cafe with good coffee and desserts")
    if query.strip():
        cleaned = clean_text(query)
        qvec = TFIDF.transform([cleaned])
        used = [w for w in dict.fromkeys(cleaned.split()) if w in VOCAB]
        unused = [w for w in dict.fromkeys(cleaned.split()) if w not in VOCAB]
        if qvec.nnz == 0:
            st.warning("None of these words appear in the restaurant reviews. Try different or simpler words.")
        else:
            st.markdown("**Words used for matching:** " + "".join(
                f"<span class='pill why'>{html.escape(readable(w, UNSTEM))}</span>" for w in used)
                + (f" &nbsp;<span class='muted'>(not found in reviews: {html.escape(', '.join(readable(w, UNSTEM) for w in unused))})</span>"
                   if unused else ""), unsafe_allow_html=True)
            recs = recommend_text(qvec, top_n, pool, MASK)
            st.subheader(f"Best matches ({len(recs)})")
            if recs.empty:
                st.info("No restaurants match these filters. Relax them or increase the similarity pool.")
            else:
                results(recs, "describe", query.strip(), qvec.tocsr())
    else:
        st.info("Type what you want - for example a vibe, a dish or an occasion - and we will search the reviews.")

# ----------------------------------------------------------------------------- tab 3: analytics
with tab_an:
    st.subheader("Bangalore restaurant market at a glance")
    k = st.columns(5)
    k[0].metric("Restaurants", f"{len(df):,}")
    k[1].metric("Locations", df["location"].nunique())
    k[2].metric("Average rating", f"{df['rate'].mean():.2f}")
    k[3].metric("Online ordering", f"{(df['online_order'] == 'Yes').mean():.0%}")
    k[4].metric("Table booking", f"{(df['book_table'] == 'Yes').mean():.0%}")

    def style(fig, h=340):
        fig.update_layout(height=h, margin=dict(l=0, r=0, t=40, b=0), title_font_size=15, showlegend=False)
        return fig

    c1, c2 = st.columns(2)
    loc = df.groupby("location")["rate"].agg(["mean", "count"]).query("count >= 20").nlargest(12, "mean").reset_index()
    c1.plotly_chart(style(px.bar(loc.iloc[::-1], x="mean", y="location", orientation="h", range_x=[3, 4.6],
                                 title="Best-rated areas (areas with 20+ restaurants)",
                                 labels={"mean": "Average rating", "location": ""}, color_discrete_sequence=[NAVY])),
                    width="stretch")
    cu = df["cuisines"].str.split(",").explode().str.strip()
    cu = cu[cu != "Unknown"].value_counts().head(12).rename_axis("cuisine").reset_index(name="restaurants")
    c2.plotly_chart(style(px.bar(cu.iloc[::-1], x="restaurants", y="cuisine", orientation="h",
                                 title="Most common cuisines", labels={"cuisine": ""}, color_discrete_sequence=[RED])),
                    width="stretch")
    c3, c4 = st.columns(2)
    band = pd.cut(df["cost"], [0, 300, 500, 800, 1200, 2000, 10000],
                  labels=["<300", "300-500", "500-800", "800-1200", "1200-2000", "2000+"])
    cb = df.groupby(band, observed=True)["rate"].mean().reset_index()
    c3.plotly_chart(style(px.bar(cb, x="cost", y="rate", range_y=[3, 4.5], title="Average rating by cost for two (₹)",
                                 labels={"cost": "Cost band", "rate": "Average rating"}, color_discrete_sequence=[NAVY])),
                    width="stretch")
    c4.plotly_chart(style(px.histogram(df, x="rate", nbins=25, title="Distribution of ratings",
                                       labels={"rate": "Rating"}, color_discrete_sequence=[RED])), width="stretch")
    st.plotly_chart(style(px.scatter(df, x="votes", y="rate", log_x=True, opacity=.35, hover_name="label",
                                     title="Votes vs rating (popular places tend to be rated higher)",
                                     labels={"votes": "Votes (log scale)", "rate": "Rating"},
                                     color_discrete_sequence=[NAVY]), 380), width="stretch")

# ----------------------------------------------------------------------------- tab 4: feedback
with tab_fb:
    st.subheader("User feedback")
    st.caption("Every 👍 / 👎 on a recommendation is saved to a local SQLite file (feedback.db). "
               "In a live deployment this is the data used to measure and improve the recommender.")
    fb = load_feedback()
    if fb.empty:
        st.info("No feedback yet - use the 👍 / 👎 buttons under any recommendation.")
    else:
        m = st.columns(3)
        m[0].metric("Votes", len(fb))
        m[1].metric("Helpful (👍)", f"{(fb['vote'] > 0).mean():.0%}")
        m[2].metric("Searches rated", fb[["mode", "query"]].drop_duplicates().shape[0])
        by_mode = fb.assign(vote=fb["vote"].map({1: "Helpful", -1: "Not helpful"})).groupby(["mode", "vote"]).size().reset_index(name="votes")
        st.plotly_chart(px.bar(by_mode, x="mode", y="votes", color="vote", barmode="group",
                               color_discrete_map={"Helpful": GREEN, "Not helpful": RED},
                               labels={"mode": "Search type"}).update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0)),
                        width="stretch")
        st.dataframe(fb.head(50), hide_index=True, width="stretch")
        st.download_button("Download feedback CSV", fb.to_csv(index=False).encode(), "feedback.csv", "text/csv")

# ----------------------------------------------------------------------------- tab 5: about
with tab_about:
    st.markdown(
        """
        ### How it works
        1. **Data** - Zomato Bangalore restaurants (8,716 with ratings and reviews after cleaning).
        2. **NLP** - all reviews of a restaurant are merged into one document, cleaned (lower-case, punctuation / URL /
           stop-word removal) and stemmed.
        3. **TF-IDF** (unigrams + bigrams) turns each document into a vector that highlights what is distinctive about
           the restaurant's reviews (dishes, ambience, service, ...).
        4. **Cosine similarity** finds restaurants whose reviews read alike - to another restaurant, or to a sentence
           you type.
        5. The closest matches are **sorted from highest rated** (votes break ties).
        6. **"Why this?"** shows the review phrases both sides have in common, so every suggestion is explainable.

        Cuisine, cost and location are *not* used for similarity - they are only optional filters.
        Map pins show the neighbourhood (approximate); the dataset has no street coordinates.
        """
    )
    a, b, c = st.columns(3)
    a.metric("Restaurants", f"{len(df):,}")
    b.metric("Locations", df["location"].nunique())
    c.metric("Average rating", f"{df['rate'].mean():.2f}")
