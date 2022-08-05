import pandas as pd
from sklearn.cluster import DBSCAN


def dbscan_features(features_df, epsilon=0.0001):
    model = DBSCAN(min_samples=1, eps=epsilon)

    clusters = pd.DataFrame(
        {"path": features_df.index, "label": model.fit_predict(features_df)}
    ).sort_values(["label", "path"], ascending=False)

    # select one of elements from each cluster to keep
    list_to_keep = set(clusters.groupby("label")["path"].first())
    all_files = set(clusters["path"])

    files_to_remove = all_files - list_to_keep
    if files_to_remove:
        print(f"Found {len(files_to_remove)} duplicate files:\n {files_to_remove} \n")

        for group, paths in clusters[clusters["label"].duplicated(keep=False)].groupby(
            "label"
        )["path"]:
            print(
                f'Duplicate items for "{paths.iloc[0]}" are:\n {paths.iloc[1:].values}\n'
            )
        return files_to_remove
    else:
        return set({})
