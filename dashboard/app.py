"""Streamlit dashboard for the Fat Loss Insights Engine.

Run with:  streamlit run dashboard/app.py

Five tabs: Profile Leaderboard, Content Mix, Top Posts, Patterns & Hooks, and
an AI-powered Insight Generator. All data is read from the SQLite DB; tabs
degrade gracefully when the pipeline hasn't been run yet.
"""
from __future__ import annotations

import json
import os
import sys

# Make the project root importable when launched via `streamlit run`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from analysis import metrics
from config import GEMINI_MODEL
from db import Profile, get_session

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

CATEGORY_COLORS = {
    "CREDIBILITY": "#1d4ed8",  # blue
    "VIRAL": "#15803d",        # green
    "LEAD_GEN": "#d97706",     # amber
    "MIXED": "#6d28d9",        # purple
}

st.set_page_config(page_title="Fat Loss Insights Engine", layout="wide")


@st.cache_data(show_spinner=False)
def load_posts_df() -> pd.DataFrame:
    with get_session() as session:
        return metrics.load_data(session)


@st.cache_data(show_spinner=False)
def load_profiles_df() -> pd.DataFrame:
    with get_session() as session:
        rows = session.query(Profile).all()
        records = [{
            "username": p.username,
            "followers": p.followers,
            "following": p.following,
            "post_count": p.post_count,
            "archetype": p.archetype,
            "avg_likes": p.avg_likes,
            "avg_comments": p.avg_comments,
            "engagement_rate": p.engagement_rate,
            "posts_per_week": p.posts_per_week,
            "relevance_score": p.relevance_score,
        } for p in rows]
    return pd.DataFrame(records)


st.title("Fat Loss Insights Engine")

with st.spinner("Loading data..."):
    df = load_posts_df()
    profiles_df = load_profiles_df()

if df.empty:
    st.warning(
        "No post data yet. Run the pipeline first: "
        "`python main.py --step all` (or each step in order)."
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Profile Leaderboard", "Content Mix", "Top Posts", "Patterns & Hooks", "Insight Generator"]
)

# --------------------------------------------------------------------------- #
# TAB 1 — Profile Leaderboard
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Profile Leaderboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total profiles", int(profiles_df["username"].nunique()) if not profiles_df.empty else 0)
    c2.metric("Total posts", int(len(df)))
    avg_er = float(df["engagement_rate"].mean()) if not df.empty else 0.0
    c3.metric("Avg engagement rate", f"{avg_er * 100:.2f}%")

    if profiles_df.empty:
        st.info("No profiles scored yet.")
    else:
        posts_per_profile = df.groupby("username").size() if not df.empty else pd.Series(dtype=int)
        table = profiles_df.copy()
        table["posts_scraped"] = table["username"].map(posts_per_profile).fillna(0).astype(int)
        table = table.sort_values("relevance_score", ascending=False)
        table["engagement_rate"] = (table["engagement_rate"].fillna(0) * 100).round(2).astype(str) + "%"
        st.dataframe(
            table[[
                "username", "followers", "archetype", "engagement_rate",
                "relevance_score", "posts_scraped",
            ]],
            use_container_width=True,
            hide_index=True,
        )

# --------------------------------------------------------------------------- #
# TAB 2 — Content Mix
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Content Mix")
    if df.empty:
        st.info("No data.")
    else:
        options = ["All profiles"] + sorted(df["username"].unique().tolist())
        selected = st.selectbox("Profile", options, key="mix_profile")

        mix = metrics.content_mix_by_profile(df)
        if selected == "All profiles":
            overall = (df["primary_category"].value_counts(normalize=True) * 100)
            mix_df = overall.rename_axis("category").reset_index(name="percent")
            fig = px.bar(
                mix_df, x="category", y="percent", color="category",
                color_discrete_map=CATEGORY_COLORS, title="Overall content mix (%)",
            )
        else:
            row = mix.loc[[selected]] if selected in mix.index else pd.DataFrame()
            mix_df = row.T.reset_index()
            mix_df.columns = ["category", "percent"]
            fig = px.bar(
                mix_df, x="category", y="percent", color="category",
                color_discrete_map=CATEGORY_COLORS, title=f"Content mix for @{selected} (%)",
            )
        st.plotly_chart(fig, use_container_width=True)

        eng = metrics.engagement_by_category(df).reset_index()
        if not eng.empty:
            fig2 = px.bar(
                eng, x="primary_category", y="mean", color="primary_category",
                color_discrete_map=CATEGORY_COLORS,
                title="Avg engagement rate by category",
            )
            st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------- #
# TAB 3 — Top Posts
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Top Posts")
    if df.empty:
        st.info("No data.")
    else:
        cat_options = ["All", "CREDIBILITY", "VIRAL", "LEAD_GEN", "MIXED"]
        media_options = ["All", "reel", "carousel", "image"]
        cc1, cc2 = st.columns(2)
        cat_filter = cc1.selectbox("Category", cat_options, key="top_cat")
        media_filter = cc2.selectbox("Media type", media_options, key="top_media")

        view = df.copy()
        if cat_filter != "All":
            view = view[view["primary_category"] == cat_filter]
        if media_filter != "All":
            view = view[view["media_type"] == media_filter]
        view = view.sort_values("likes", ascending=False).head(20)

        for _, post in view.iterrows():
            cat = post.get("primary_category") or "UNKNOWN"
            header = (
                f"@{post['username']}  ·  ❤ {int(post['likes'])}  "
                f"💬 {int(post['comments'])}  👁 {int(post.get('views') or 0)}  ·  {cat}"
            )
            with st.expander(header):
                if post.get("hook"):
                    st.markdown(f"**Hook:** {post['hook']}")
                if post.get("cta_text") and str(post["cta_text"]).lower() != "none":
                    st.markdown(f":orange[**CTA:** {post['cta_text']}]")
                st.markdown(f"[Open original post]({post['post_url']})")

# --------------------------------------------------------------------------- #
# TAB 4 — Patterns & Hooks
# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("Patterns & Hooks")
    if df.empty:
        st.info("No data.")
    else:
        st.markdown("### Most common CTAs")
        ctas = metrics.cta_pattern_frequency(df).head(15)
        if not ctas.empty:
            cta_df = ctas.rename_axis("cta").reset_index(name="count")
            fig = px.bar(cta_df, x="count", y="cta", orientation="h", title="Top CTA phrases")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No CTA data.")

        st.markdown("### Lead Gen Sequence")
        seq = metrics.lead_gen_sequence(df)
        if not seq.empty:
            prev_freq = (
                seq["prev_category"].value_counts(normalize=True) * 100
            ).round(1).rename_axis("prev_category").reset_index(name="percent")
            st.dataframe(prev_freq, use_container_width=True, hide_index=True)
        else:
            st.caption("No lead-gen posts found.")

        st.markdown("### Top Hooks by Category")
        for cat in ["CREDIBILITY", "VIRAL", "LEAD_GEN", "MIXED"]:
            hooks = metrics.top_hooks(df, cat, 10)
            if hooks:
                st.markdown(f"**{cat}**")
                for h in hooks:
                    st.markdown(f"- {h}")

        st.markdown("### Format vs. Engagement")
        fmt = metrics.format_vs_engagement(df)
        if not fmt.empty:
            st.dataframe((fmt * 100).round(3), use_container_width=True)

# --------------------------------------------------------------------------- #
# TAB 5 — Insight Generator
# --------------------------------------------------------------------------- #
with tab5:
    st.subheader("Insight Generator")
    summary = metrics.generate_insights_summary(df)
    summary_json = json.dumps(summary, indent=2, default=str)
    st.code(summary_json, language="json")

    if st.button("Generate Strategy Recommendations"):
        if not GEMINI_API_KEY:
            st.error("GEMINI_API_KEY is not set in .env.")
        else:
            with st.spinner("Generating recommendations..."):
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=GEMINI_API_KEY)
                    model = genai.GenerativeModel(GEMINI_MODEL)
                    prompt = (
                        "You are a health coaching content strategist. Based on this "
                        f"data from analyzing {summary['total_profiles']} top fat loss "
                        f"influencers and {summary['total_posts']} posts, give 5 specific, "
                        "actionable content strategy recommendations for a health coaching "
                        f"platform.\nData: {summary_json}\n"
                        "Format each recommendation as: [Title] — [What to do] — [Why it works]"
                    )
                    response = model.generate_content(prompt)
                    st.write(response.text)
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")
