import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def Silhouette_DBI_three_category(p1, p2, p3, p4, knob):
    csv_path = f"./categorization/all_benchmarks-difficulty_{p1}_{p2}_{p3}_{p4}@{knob}.csv"
    df = pd.read_csv(csv_path)

    metric_cols = [c for c in df.columns if c not in ['difficulty', 'index', 'dataset']]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    df[metric_cols] = df[metric_cols].astype(float).round(12)
    X_scaled = StandardScaler().fit_transform(df[metric_cols])
    label_mapping = {"easy": 0, "medium": 1, "difficult": 2}
    df["label"] = df["difficulty"].map(label_mapping)

    present_labels = sorted(df["label"].dropna().unique())
    label_name_from_id = {0: "easy", 1: "medium", 2: "difficult"}

    if len(present_labels) < 2:
        print(":warning: Only one difficulty class present; silhouette and DBI are undefined.")
        print(df["difficulty"].value_counts())
    else:
        db_index = davies_bouldin_score(X_scaled, df["label"])
        print(f"Davies–Bouldin Index: {db_index:.3f}")

        sample_scores = silhouette_samples(X_scaled, df["label"])
        df["silhouette_score"] = sample_scores

        df["flag"] = df["silhouette_score"].apply(
            lambda s: ":triangular_flag_on_post: borderline" if (-0.15 < s < 0.15) else ""
        )

        overall_score = silhouette_score(X_scaled, df["label"])
        print(f":mag_right: Overall Silhouette Score: {overall_score:.3f}")
        if overall_score > 0.5:
            print(":white_check_mark: Clusters are well-separated. (silhouette)")
        elif overall_score > 0.25:
            print(":warning: Moderate cluster separation. Check borderline cases (silhouette).")
        else:
            print(":x: Poor separation. Consider revising categorization (silhouette).")

        if db_index < 0.5:
            print(":white_check_mark: Clusters are compact and well-separated (DBI).")
        elif db_index < 1.0:
            print(":warning: Clusters are moderately separated (DBI).")
        else:
            print(":x: Poor separation. Clusters overlap (DBI).")

        print("\n:rotating_light: Red-flagged borderline programs:")
        cols_to_show = ["difficulty", "silhouette_score", "flag"]
        print(df.loc[df["flag"] != "", cols_to_show])

        fig, ax = plt.subplots(figsize=(8, 6))
        y_lower = 10
        colors = {0: "limegreen", 1: "gold", 2: "tomato"}

        for i in [0, 1, 2]:
            if i not in present_labels:
                continue
            ith_scores = df.loc[df["label"] == i, "silhouette_score"].to_numpy()
            ith_scores.sort()
            size = len(ith_scores)
            ax.fill_betweenx(np.arange(y_lower, y_lower + size), 0, ith_scores,
                            facecolor=colors[i], alpha=0.7)
            ax.text(-0.1, y_lower + 0.5 * size, label_name_from_id[i])
            y_lower += size + 10

        ax.axvline(x=overall_score, color="blue", linestyle="--")
        ax.set_title("Silhouette Plot for Difficulty Clustering")
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Samples")
        plt.tight_layout()
        plt.savefig(f"./categorization/Silhouette_3_{p1}_{p2}_{p3}_{p4}@{knob}.jpeg")


def Silhouette_DBI_two_category(p1, p2, p3, p4, knob, remove_borderlines=False):
    csv_path = f"./categorization/all_benchmarks-difficulty_{p1}_{p2}_{p3}_{p4}@{knob}.csv"
    df = pd.read_csv(csv_path)
    df = df[df["difficulty"] != "medium"] ## remove row marked as "medium"
    metric_cols = [c for c in df.columns if c not in ['difficulty', 'index', 'dataset']]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    df[metric_cols] = df[metric_cols].astype(float).round(12)
    X_scaled = StandardScaler().fit_transform(df[metric_cols])
    
    label_mapping = {"easy": 0, "difficult": 1}
    df["label"] = df["difficulty"].map(label_mapping)

    present_labels = sorted(df["label"].dropna().unique())
    label_name_from_id = {0: "easy", 1: "difficult"}

    if len(present_labels) < 2:
        print(":warning: Only one difficulty class present; silhouette and DBI are undefined.")
        print(df["difficulty"].value_counts())
    else:
        db_index = davies_bouldin_score(X_scaled, df["label"])
        print(f"Davies–Bouldin Index: {db_index:.3f}")

        sample_scores = silhouette_samples(X_scaled, df["label"])
        df["silhouette_score"] = sample_scores

        df["flag"] = df["silhouette_score"].apply(
            lambda s: ":triangular_flag_on_post: borderline" if (-0.15 < s < 0.15) else ""
        )

        overall_score = silhouette_score(X_scaled, df["label"])
        # print(f":mag_right: Overall Silhouette Score: {overall_score:.3f}")
        # if overall_score > 0.5:
        #     print(":white_check_mark: Clusters are well-separated. (silhouette)")
        # elif overall_score > 0.25:
        #     print(":warning: Moderate cluster separation. Check borderline cases (silhouette).")
        # else:
        #     print(":x: Poor separation. Consider revising categorization (silhouette).")

        # if db_index < 0.5:
        #     print(":white_check_mark: Clusters are compact and well-separated (DBI).")
        # elif db_index < 1.0:
        #     print(":warning: Clusters are moderately separated (DBI).")
        # else:
        #     print(":x: Poor separation. Clusters overlap (DBI).")

        cols_to_show = ["difficulty", "silhouette_score", "flag"]
        
        if remove_borderlines:
            borderline_programs = df.loc[df["flag"] != "", "index"].tolist()
            return borderline_programs, db_index, overall_score
        else:
            return [] , db_index, overall_score
