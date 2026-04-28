"""Data loading, vocabulary construction, and train/test splitting."""
# Import Dependencies
from datasets import load_dataset

from dataclasses import dataclass


import os
import zipfile
import urllib.request
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# from config import DATA_URL, DATA_DIR, GENRE_COLS, N_GENRES, SEED


# Download & load
def _get_data():
    data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
    print(data["full"][0])
    return data



# def download_data():
#     """Download and extract MovieLens 100K if not already present."""
#     if not os.path.exists(DATA_DIR):
#         print("Downloading MovieLens 100K …")
#         urllib.request.urlretrieve(DATA_URL, "ml-100k.zip")
#         with zipfile.ZipFile("ml-100k.zip") as z:
#             z.extractall(".")
#         print("Done.")


# def load_ratings():
#     """Load ratings and item metadata, return merged DataFrame."""
#     ratings = pd.read_csv(
#         f"{DATA_DIR}/u.data", sep="\t",
#         names=["user", "movie", "rating", "ts"],
#     )
#     titles = pd.read_csv(
#         f"{DATA_DIR}/u.item", sep="|", encoding="latin-1",
#         header=None, usecols=[0, 1] + list(range(5, 24)),
#         names=["movie", "title"] + GENRE_COLS,
#     )
#     return ratings.merge(titles, on="movie"), titles


# # ── Vocabularies ─────────────────────────────────────────────────────────

# def build_vocabs(ratings):
#     """Build user/movie index mappings and per-user positive-item sets."""
#     users  = sorted(ratings["user"].unique())
#     movies = sorted(ratings["title"].unique())

#     u2i = {u: i for i, u in enumerate(users)}
#     m2i = {m: i for i, m in enumerate(movies)}
#     i2m = {i: m for m, i in m2i.items()}

#     user_pos = {}
#     for _, row in ratings.iterrows():
#         uid = u2i[row["user"]]
#         mid = m2i[row["title"]]
#         user_pos.setdefault(uid, set()).add(mid)

#     return u2i, m2i, i2m, user_pos


# def build_genre_tensor(titles, m2i, device):
#     """Build (N_MOVIES, N_GENRES) float tensor aligned to movie indices."""
#     n_movies = len(m2i)
#     title_to_genre = titles.drop_duplicates(subset="title").set_index("title")[GENRE_COLS]
#     genre_tensor = torch.zeros(n_movies, N_GENRES)
#     for title, idx in m2i.items():
#         if title in title_to_genre.index:
#             genre_tensor[idx] = torch.tensor(
#                 title_to_genre.loc[title].values, dtype=torch.float32
#             )
#     return genre_tensor.to(device)


# # ── Split ────────────────────────────────────────────────────────────────

# def train_test_split(ratings, train_size=80_000):
#     """Random 80k/20k split."""
#     perm = np.random.permutation(len(ratings))
#     train_df = ratings.iloc[perm[:train_size]].reset_index(drop=True)
#     test_df  = ratings.iloc[perm[train_size:]].reset_index(drop=True)
#     return train_df, test_df


# # ── Dataset ──────────────────────────────────────────────────────────────

# class BPRDataset(Dataset):
#     """Yields (user_idx, pos_movie_idx, pos_genre, neg_movie_idx, neg_genre)."""

#     def __init__(self, df, u2i, m2i, user_pos, movie_genres_cpu, n_movies):
#         self.pairs = [(u2i[r["user"]], m2i[r["title"]]) for _, r in df.iterrows()]
#         self.genres = movie_genres_cpu
#         self.user_pos = user_pos
#         self.n_movies = n_movies

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, i):
#         u, pos = self.pairs[i]
#         neg = random.randint(0, self.n_movies - 1)
#         while neg in self.user_pos[u]:
#             neg = random.randint(0, self.n_movies - 1)
#         return u, pos, self.genres[pos], neg, self.genres[neg]
