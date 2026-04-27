import plotly.express as px
import pandas as pd


def score_bar_chart(top_df):
    fig = px.bar(
        top_df,
        x="match_score",
        y=top_df.index.astype(str),
        orientation="h",
        color="match_score",
        text="match_score",
        title="Top Resume Match Scores"
    )

    fig.update_layout(
        yaxis_title="Resume Rank",
        xaxis_title="Match Score (%)",
        yaxis={"autorange": "reversed"}
    )

    return fig


def category_pie(df):
    counts = df["Category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    fig = px.pie(
        counts,
        names="Category",
        values="Count",
        title="Resume Categories"
    )

    return fig


def generate_wordcloud_img(text):
    # fallback text summary instead of image
    words = text.split()
    freq = pd.Series(words).value_counts().head(30)

    fig = px.bar(
        x=freq.values,
        y=freq.index,
        orientation="h",
        title="Top Keywords in Matching Resumes"
    )

    fig.update_layout(yaxis={"autorange": "reversed"})

    return fig