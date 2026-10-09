"""KDD Project: Math Problem Topic Classifier (Streamlit dashboard).

Run locally:   streamlit run app.py
Needs ML_PROJECT.csv in the same folder as this file.
"""
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

CSV_PATH = Path(__file__).with_name("ML_PROJECT.csv")
MODEL_ORDER = ["Logistic Regression", "Linear SVM", "Naive Bayes", "Random Forest"]
EXAMPLES = {
    "Geometry": "Find the area of a triangle with sides 5, 12 and 13.",
    "Number Theory": "What is the remainder when 7^100 is divided by 13?",
    "Counting & Probability": "In how many ways can 5 people sit in a row of 5 chairs?",
    "Algebra": "Solve for x: 3x + 7 = 22.",
    "Precalculus": "Find the angle between vectors a and b if cos theta equals the dot product divided by their magnitudes.",
    "Intermediate Algebra": "Find the number of real roots of x^4 - 5x^2 + 4 = 0 and the sum of all roots of the polynomial.",
}


# ---------------------------------------------------------------- KDD pipeline
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"\$+", " ", t)        # remove $ signs
    t = re.sub(r"\d+", " num ", t)    # replace numbers
    t = re.sub(r"[^a-z\s]", " ", t)   # remove symbols
    return re.sub(r"\s+", " ", t).strip()


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@st.cache_resource(show_spinner="Training the four models. This happens once and takes about a minute...")
def train_all():
    raw = pd.read_csv(CSV_PATH)
    # 1. Cleaning
    df = raw.drop_duplicates(subset="problem").dropna(subset=["problem", "solution"])
    df = df[df["level"] != "Level ?"].copy()
    df["problem_clean"] = df["problem"].apply(clean_text)
    # 2. Selection (only rows with a topic label)
    data = df.dropna(subset=["type"]).copy()
    # 3. Transformation
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_features=20000)
    X = tfidf.fit_transform(data["problem_clean"])
    y = data["type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    labels = sorted(y.unique())
    # 4. Data mining
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5),
        "Linear SVM": LinearSVC(C=0.5),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    }
    rows, preds = [], {}
    for name in MODEL_ORDER:
        m = models[name]
        t0 = time.time()
        m.fit(X_train, y_train)
        train_time = time.time() - t0
        p = m.predict(X_test)
        preds[name] = p
        cv = cross_val_score(m, X, y, cv=5, scoring="accuracy")
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, p),
                     "Precision": precision_score(y_test, p, average="macro"),
                     "Recall": recall_score(y_test, p, average="macro"),
                     "F1": f1_score(y_test, p, average="macro"),
                     "CV Accuracy": cv.mean(), "CV Std": cv.std(),
                     "Train Time (s)": train_time})
    results = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best = results.loc[0, "Model"]
    # 5. Interpretation
    lr = models["Logistic Regression"]
    terms = tfidf.get_feature_names_out()
    top_words = {c: list(terms[lr.coef_[i].argsort()[::-1][:8]]) for i, c in enumerate(lr.classes_)}
    report = pd.DataFrame(classification_report(y_test, preds[best], output_dict=True)).T
    cm = confusion_matrix(y_test, preds[best], labels=labels)
    # Extra: clustering
    km = KMeans(n_clusters=7, n_init=10, random_state=42)
    ari = adjusted_rand_score(y, km.fit_predict(X))
    return dict(raw_rows=len(raw), clean_rows=len(df), math_rows=len(data), gsm_rows=int(df["type"].isna().sum()),
                dups=int(len(raw) - raw["problem"].nunique()), types=data["type"].value_counts(),
                levels=data["level"].value_counts().sort_index(), tfidf=tfidf, models=models,
                results=results, best=best, labels=labels, report=report, cm=cm, top_words=top_words,
                ari=ari, n_test=len(y_test), terms=terms)


def predict_all(texts, B):
    """Return the TF-IDF matrix and a probability table for every algorithm."""
    X = B["tfidf"].transform([clean_text(t) for t in texts])
    M = B["models"]
    probs = {
        "Logistic Regression": M["Logistic Regression"].predict_proba(X),
        "Linear SVM": softmax(M["Linear SVM"].decision_function(X) * 4),  # rescaled margin, not a true probability
        "Naive Bayes": M["Naive Bayes"].predict_proba(X),
        "Random Forest": M["Random Forest"].predict_proba(X),
    }
    return X, probs


def explain_best(B, probs_row, tops):
    """Plain-English analysis of the most accurate model for one problem."""
    best, labels, cls = B["best"], B["labels"], list(B["models"][B["best"]].classes_)
    acc = B["results"].loc[0, "Accuracy"]
    p = probs_row[best]
    order = np.argsort(p)[::-1]
    topic, conf = cls[order[0]], p[order[0]]
    runner, rconf = cls[order[1]], p[order[1]]
    same = [m for m in MODEL_ORDER if tops[m] == topic]
    diff = [m for m in MODEL_ORDER if tops[m] != topic]
    rep = B["report"].loc[topic]
    j = labels.index(topic)
    col = B["cm"][:, j].copy()
    col[j] = 0
    lines = [f"**{best}** has the highest test accuracy of the four ({acc:.1%}), so its answer is the headline result: "
             f"**{topic}** at {conf:.1%} confidence."]
    if len(same) == 4:
        lines.append(f"All four algorithms chose {topic}. Different methods reaching the same answer makes this prediction more trustworthy.")
    else:
        agree = [m for m in same if m != best]
        txt = (" and ".join(agree) + f" also chose {topic}. ") if agree else f"No other algorithm chose {topic}. "
        txt += "; ".join(f"{m} chose {tops[m]} ({probs_row[m][cls.index(tops[m])]:.1%})" for m in diff) + "."
        lines.append(txt)
    rel = (f"On the test set, when {best} predicts {topic} it is right {rep['precision']:.1%} of the time, "
           f"and it finds {rep['recall']:.1%} of real {topic} problems.")
    if col.max() > 0:
        rel += f" Problems it labels {topic} are most often really {labels[int(col.argmax())]} ({int(col.max())} of {int(B['cm'][:, j].sum())})."
    lines.append(rel)
    if conf - rconf < 0.2:
        lines.append(f"This is a close call: the runner-up {runner} has {rconf:.1%}. Read it as a lean, not a certainty.")
    elif conf < 0.5:
        lines.append("Confidence is under 50%, so treat this as a best guess.")
    else:
        lines.append(f"The gap to the runner-up ({runner}, {rconf:.1%}) is wide, so the model is fairly decisive.")
    return topic, conf, lines


def evidence_words(B, X_row, topic, n=8):
    lr = B["models"]["Logistic Regression"]
    k = list(lr.classes_).index(topic)
    row = X_row.tocoo()
    contrib = sorted(((B["terms"][c], v * lr.coef_[k][c]) for c, v in zip(row.col, row.data)), key=lambda a: -a[1])
    return [w for w, s in contrib if s > 0][:n]


def analyse_batch(texts, B):
    X, probs = predict_all(texts, B)
    cls = list(B["models"][B["best"]].classes_)
    out = pd.DataFrame({"problem": texts})
    for m in MODEL_ORDER:
        out[f"{m} topic"] = [cls[i] for i in probs[m].argmax(axis=1)]
    for m in MODEL_ORDER:
        out[f"{m} confidence"] = probs[m].max(axis=1).round(4)
    out["best model answer"] = out[f"{B['best']} topic"]
    out["models agreeing"] = [sum(out.loc[i, f"{m} topic"] == out.loc[i, "best model answer"] for m in MODEL_ORDER)
                              for i in range(len(out))]
    out.loc[np.asarray(X.getnnz(axis=1)) == 0, ["best model answer"]] = "unknown (no recognised words)"
    return out


# ------------------------------------------------------------------------- UI
def main():
    st.set_page_config(page_title="Math Problem Topic Classifier", page_icon="📐", layout="wide")
    st.title("Which topic is this math problem about?")
    if not CSV_PATH.exists():
        st.error("ML_PROJECT.csv was not found next to app.py. Add it to the same folder and rerun.")
        st.stop()
    B = train_all()
    best, res = B["best"], B["results"]
    cls = list(B["models"][best].classes_)

    st.caption(f"KDD project on {B['math_rows']:,} labelled MATH problems. Predict a new problem with all four algorithms, "
               f"then read the analysis of the most accurate one ({best}, {res.loc[0, 'Accuracy']:.1%}).")
    t_pred, t_batch, t_report, t_results, t_data = st.tabs(
        ["Predict", "Batch", "Project report", "Results", "Data and words"])

    with t_pred:
        ex = st.selectbox("Try an example (or write your own below)", ["(none)"] + list(EXAMPLES.values()))
        text = st.text_area("Math problem", value="" if ex == "(none)" else ex, height=120,
                            placeholder="Type or paste a math problem here...")
        if st.button("Analyze problem", type="primary"):
            if not text.strip():
                st.warning("Enter a problem first.")
            else:
                X, probs = predict_all([text], B)
                if X.nnz < 2:
                    st.warning("Too few recognised words. Write the problem in full English sentences.")
                else:
                    row = {m: probs[m][0] for m in MODEL_ORDER}
                    tops = {m: cls[int(row[m].argmax())] for m in MODEL_ORDER}
                    topic, conf, lines = explain_best(B, row, tops)
                    st.subheader(f"{topic}  ({conf:.1%} confident, {best})")
                    st.markdown(f"**{sum(t == topic for t in tops.values())} of 4 algorithms agree with the best model.**")
                    st.markdown("##### All four algorithms")
                    acc = res.set_index("Model")["Accuracy"]
                    tbl = pd.DataFrame([{"Algorithm": m + (" (best)" if m == best else ""),
                                         "Test accuracy": acc[m], "Predicted topic": tops[m],
                                         "Confidence": row[m].max(),
                                         "Versus best": "-" if m == best else ("agrees" if tops[m] == topic else "differs")}
                                        for m in res["Model"]])
                    st.dataframe(tbl.style.format({"Test accuracy": "{:.1%}", "Confidence": "{:.1%}"}),
                                 hide_index=True, use_container_width=True)
                    st.caption("SVM has no true probabilities; its confidence is a rescaled margin.")
                    st.markdown("##### Probability of each topic, by algorithm")
                    heat = pd.DataFrame({m: row[m] for m in MODEL_ORDER}, index=cls)
                    st.dataframe(heat.style.background_gradient(cmap="Blues", axis=None, vmin=0, vmax=1).format("{:.1%}"),
                                 use_container_width=True)
                    st.markdown("##### Analysis of the best model")
                    for ln in lines:
                        st.markdown(ln)
                    words = evidence_words(B, X[0], topic)
                    if words:
                        st.markdown("Words that pushed toward the predicted topic: " + ", ".join(f"`{w}`" for w in words))

    with t_batch:
        st.write("Analyze many problems at once: one per line, or upload a CSV with a `problem` column.")
        bt = st.text_area("Problems (one per line)", height=140, key="batch_text")
        up = st.file_uploader("Or upload a CSV", type="csv")
        if st.button("Analyze batch"):
            texts = [s.strip() for s in bt.splitlines() if s.strip()]
            if up is not None:
                d = pd.read_csv(up)
                if "problem" in d.columns:
                    texts += d["problem"].dropna().astype(str).tolist()
                else:
                    st.error("The CSV needs a column named `problem`.")
            texts = texts[:2000]
            if texts:
                out = analyse_batch(texts, B)
                st.dataframe(out, use_container_width=True)
                n = len(out)
                st.markdown(f"All 4 algorithms agree on **{(out['models agreeing'] == 4).sum()} of {n}** problems "
                            f"({(out['models agreeing'] == 4).mean():.1%}).")
                c1, c2 = st.columns(2)
                c1.markdown(f"Topics found by {best}")
                c1.bar_chart(out["best model answer"].value_counts())
                agr = {m: (out[f"{m} topic"] == out[f"{best} topic"]).mean() for m in MODEL_ORDER if m != best}
                c2.markdown(f"Agreement with {best}")
                c2.bar_chart(pd.Series(agr))
                st.download_button("Download results (CSV)", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
            else:
                st.warning("Add at least one problem.")

    with t_report:
        st.markdown(f"""
**Goal:** predict the topic of a math problem from its text. **Data:** `ML_PROJECT.csv` (MATH and GSM8K problems; columns problem, level, type, solution, stage, source).

**1. Cleaning and preprocessing.** {B['raw_rows']:,} rows loaded. Removed {B['dups']} duplicate problems, rows with missing problem or solution, and the invalid level "Level ?". Text is lowercased, `$` signs removed, numbers replaced by the token `num`, symbols removed. Result: {B['clean_rows']:,} rows.

**2. Data selection.** Only rows with a topic are kept. GSM8K rows have no type, so {B['gsm_rows']:,} rows are dropped, leaving {B['math_rows']:,} MATH problems across 7 topics: {", ".join(B['labels'])}.

**3. Data transformation.** TF-IDF (English stop words, 1 and 2 word phrases, minimum 3 occurrences, 20,000 features). 80% training and 20% testing ({B['n_test']:,} test problems), stratified by topic, `random_state=42`.

**4. Data mining.** Four algorithms, each tested on unseen problems and checked with 5-fold cross-validation:
- Logistic Regression (`C=5`, `max_iter=1000`): weights each word per topic
- Linear SVM (`C=0.5`): widest-margin straight boundary
- Naive Bayes (`alpha=0.1`): word probabilities per topic
- Random Forest (200 trees): majority vote of decision trees

**5. Interpretation and evaluation.** Models are compared on accuracy, macro precision, recall and F1, CV accuracy and training time. Per-topic scores, the confusion matrix and top words per topic explain what the best model learned.

**Extra: clustering.** KMeans with 7 clusters groups problems without labels. Adjusted Rand Index: **{B['ari']:.3f}**, so word similarity alone does not recover the topics.
""")

    with t_results:
        st.subheader("Model comparison")
        show = res.rename(columns={"CV Std": "CV +/-"})
        st.dataframe(show.style.format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1", "CV Accuracy", "CV +/-"]} | {"Train Time (s)": "{:.2f}"}),
                     hide_index=True, use_container_width=True)
        st.bar_chart(res.set_index("Model")[["Accuracy", "F1", "CV Accuracy"]])
        st.subheader(f"Per-topic scores ({best})")
        st.dataframe(B["report"].round(3), use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(B["cm"], annot=True, fmt="d", cmap="Blues", xticklabels=B["labels"], yticklabels=B["labels"], ax=ax)
        ax.set_title(f"Confusion matrix - {best}")
        ax.set_xlabel("Predicted topic")
        ax.set_ylabel("Actual topic")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        f1 = B["report"].loc[B["labels"], "f1-score"].sort_values()
        st.subheader("Conclusion")
        st.markdown(f"All four algorithms predict the topic correctly about {res['Accuracy'].min():.0%} to {res['Accuracy'].max():.0%} of the time. "
                    f"{best} scored highest ({res.loc[0, 'Accuracy']:.1%}); {res.loc[3, 'Model']} was lowest ({res.loc[3, 'Accuracy']:.1%}). "
                    f"{f1.index[0]} is the hardest topic (F1 {f1.iloc[0]:.2f}) and {f1.index[-1]} the easiest (F1 {f1.iloc[-1]:.2f}). "
                    f"Clustering does not match the real topics well (Adjusted Rand Index {B['ari']:.2f}). "
                    "Cross-validation scores sit close to the test scores, so the results are stable.")

    with t_data:
        st.write(f"{B['raw_rows']:,} rows in the file. {B['dups']} duplicates and {B['gsm_rows']:,} GSM8K rows without a topic were removed, "
                 f"leaving {B['math_rows']:,} labelled MATH problems.")
        c1, c2 = st.columns(2)
        c1.markdown("Problems per topic")
        c1.bar_chart(B["types"])
        c2.markdown("Problems per difficulty level")
        c2.bar_chart(B["levels"])
        st.subheader("Words the model links to each topic")
        st.caption('Highest-weight words from Logistic Regression. "num" stands for any number in the text.')
        st.dataframe(pd.DataFrame({k: [", ".join(v)] for k, v in B["top_words"].items()}).T.rename(columns={0: "Top words"}),
                     use_container_width=True)


if __name__ == "__main__":
    main()
