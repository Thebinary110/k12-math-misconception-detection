"""How much is each signal worth if read perfectly? Uses gold train fields (not available at test) as an upper bound:
prior + answer-correct flag, and TF-IDF/LR on the gold transcription (annotator notes removed). Out-of-fold MAP@3."""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from answer_parsing import strip_notes
from hierarchical_prior import fit, predict
from common import classes, folds, load, map3, top3


def main():
    train, _, taxonomy = load()
    cls = classes(train, taxonomy)
    idx = {c: i for i, c in enumerate(cls)}
    y = train["root_code"].values
    yi = train["root_code"].map(idx).values
    none = idx["NONE"]
    flag = train["final_answer_correct"].astype(bool).values
    text = [strip_notes(t) for t in train["transcription_steps"]]
    text = [q + "\n" + t for q, t in zip(train["question_text"], text)]
    splits = folds(train)

    P_prior = np.zeros((len(train), len(cls)))
    P_lr = np.zeros_like(P_prior)
    for tr_idx, va_idx in splits:
        P_prior[va_idx] = predict(train.iloc[va_idx], fit(train.iloc[tr_idx], cls, 8, 30))
        w = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        Xtr = w.fit_transform([text[i] for i in tr_idx])
        Xva = w.transform([text[i] for i in va_idx])
        Xtr = hstack([Xtr, csr_matrix(flag[tr_idx, None] * 3.0)]).tocsr()
        Xva = hstack([Xva, csr_matrix(flag[va_idx, None] * 3.0)]).tocsr()
        clf = LogisticRegression(C=4.0, max_iter=3000).fit(Xtr, yi[tr_idx])
        P_lr[va_idx][:, :] = 0
        tmp = np.zeros((len(va_idx), len(cls)))
        tmp[:, clf.classes_] = clf.predict_proba(Xva)
        P_lr[va_idx] = tmp

    print(f"prior only                         {map3(y, top3(P_prior, cls)):.4f}")
    # prior + perfect answer-correct flag: correct -> NONE strongly, wrong -> renormalise without NONE
    P_flag = P_prior.copy()
    P_flag[flag, none] += 5.0
    P_flag[~flag, none] *= 0.02
    print(f"prior + gold answer-correct flag   {map3(y, top3(P_flag, cls)):.4f}")
    print(f"TF-IDF/LR on gold text + flag      {map3(y, top3(P_lr, cls)):.4f}")
    for a in (0.3, 0.5, 0.7):
        print(f"blend {a:.1f}*LR + prior-with-flag   {map3(y, top3(a * P_lr + (1 - a) * P_flag / P_flag.sum(1, keepdims=True), cls)):.4f}")


if __name__ == "__main__":
    main()
