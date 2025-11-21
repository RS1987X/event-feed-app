# filename: test_semantic_similarity.py
# pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer, util
import numpy as np

# ----------------------------
# 1) HARD-CODED MODEL
# ----------------------------
MODEL_NAME = "intfloat/multilingual-e5-base"
model = SentenceTransformer(MODEL_NAME)

# ----------------------------
# 2) PASTE YOUR TITLES HERE (all-vs-all mode)
#    Fill this list with any headlines/titles you want to compare.
# ----------------------------
titles = [
    "CEO resignation",
    "VD avgår",
    "CEO steps down",
    "VD lämnar sin post",
    "CFO resignation",
    "VD tillträder",
    # add/remove as you like...
]

# ----------------------------
# 3) OPTIONAL: PASTE EXPLICIT PAIRS HERE (pair mode)
#    Each tuple is (text_a, text_b).
# ----------------------------
pairs = [
    ("Aktia Bank Abp: Anmälan om ändring av ägarandel i enlighet med kapitel 9 paragraf 10 i värdepappersmarknadslagen", "Aktia Pankki Oyj: Ilmoitus omistusosuuden muutoksesta arvopaperimarkkinalain 9. luvun 10. pykälän mukaisesti (RG Partners Oy)"),
    ("CEO steps down", "VD lämnar sin post"),
    ("Hoist Finance expands into Finland", "Product/service launches and upgrades; partnerships, collaborations, joint ventures, pilot programs."),
    ("Welcome as subscriber at Finnair From now on, you will receive information from Finnair to this e-mail address. To unsubscribe, please click on the link below.(If you can't click on the link, copy the link","Forward-looking statements: guidance initiations/updates/withdrawals; trading updates; profit warnings.")
    
]

def print_similarity_matrix(texts, normalize=True):
    if not texts:
        print("[all-vs-all] No texts provided.")
        return
    print("\n=== All-vs-All Cosine Similarity Matrix ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Texts ({len(texts)}): {texts}\n")

    # Encode and (optionally) L2-normalize embeddings
    embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=normalize)
    # cosine since normalized => dot product
    sim = (embs @ embs.T).cpu().numpy()

    # pretty print
    max_len = max(len(t) for t in texts)
    col_w = max(12, min(40, max_len + 2))
    header = " " * col_w + "".join(f"{i:>10}" for i in range(len(texts)))
    print(header)
    for i, row in enumerate(sim):
        name = texts[i][:col_w-2] + ("…" if len(texts[i]) > col_w-2 else "")
        print(f"{name:<{col_w}}" + "".join(f"{v:>10.3f}" for v in row))
    print("\nGuide: 1.0 = identical, >0.80 very close, 0.60–0.80 related, <0.60 dissimilar")

def print_pair_scores(pairs, normalize=True):
    if not pairs:
        print("[pairs] No pairs provided.")
        return
    print("\n=== Pairwise Cosine Similarity ===")
    print(f"Model: {MODEL_NAME}\n")
    a_texts = [a for a, _ in pairs]
    b_texts = [b for _, b in pairs]

    a_embs = model.encode(a_texts, convert_to_tensor=True, normalize_embeddings=normalize)
    b_embs = model.encode(b_texts, convert_to_tensor=True, normalize_embeddings=normalize)

    sims = util.cos_sim(a_embs, b_embs).cpu().numpy()
    for i, (a, b) in enumerate(pairs):
        print(f"'{a}'  vs  '{b}'  ->  cos_sim = {sims[i, i]:.3f}")
    print("\nGuide: 1.0 = identical, >0.80 very close, 0.60–0.80 related, <0.60 dissimilar")

if __name__ == "__main__":
    # Run both modes; comment out what you don't need.
    #print_similarity_matrix(titles, normalize=True)
    print_pair_scores(pairs, normalize=True)
