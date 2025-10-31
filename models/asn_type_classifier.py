from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, OneHotEncoder


@dataclass
class DatasetPaths:
    peeringdb_all: Path
    asrank_features: Path
    domain_counts: Path
    geo_aggregated: Path


def _load_peeringdb(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"aut": "asn", "label": "info_type"})
    df = df[df["info_type"].notna()].copy()
    df["asn"] = df["asn"].astype(int)
    df["org_name"] = df["org_name"].fillna("")
    return df


def _load_asrank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"asn": "asn"})
    feature_map = {
        col: f"asrank_{col}"
        for col in df.columns
        if col != "asn"
    }
    df = df.rename(columns=feature_map)
    return df


def _load_domain_counts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"ASN": "asn", "domains": "domain_count"})
    df["domain_count"] = df["domain_count"].astype(float)
    return df


def _gini(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    total = values.sum()
    if total <= 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_vals)) / (n * total) - (n + 1) / n)


def _compute_geo_features(group: pd.DataFrame) -> pd.Series:
    cleaned = group.dropna(subset=["latitude", "longitude", "address_count"])
    if cleaned.empty:
        return pd.Series(
            {
                "geo_total_addresses": np.nan,
                "geo_num_locations": 0,
                "geo_num_countries": 0,
                "geo_top_country_iso": "UNKNOWN",
                "geo_top_country_share": 0.0,
                "geo_top3_country_share": 0.0,
                "geo_country_entropy": 0.0,
                "geo_country_gini": 0.0,
                "geo_weighted_latitude": np.nan,
                "geo_weighted_longitude": np.nan,
                "geo_latitude_std": np.nan,
                "geo_longitude_std": np.nan,
            }
        )

    cleaned = cleaned.copy()
    cleaned["country_iso"] = cleaned["country"].str.split(".").str[0]
    weight_per_iso = (
        cleaned.groupby("country_iso")["address_count"]
        .sum()
        .sort_values(ascending=False)
    )

    weights = cleaned["address_count"].to_numpy(dtype=float)
    total_addresses = weights.sum()
    if total_addresses <= 0:
        weights = np.ones_like(weights)
        total_addresses = weights.sum()

    latitude = cleaned["latitude"].to_numpy(dtype=float)
    longitude = cleaned["longitude"].to_numpy(dtype=float)

    weighted_lat = float(np.average(latitude, weights=weights))
    weighted_lon = float(np.average(longitude, weights=weights))

    lat_var = float(np.average((latitude - weighted_lat) ** 2, weights=weights))
    lon_var = float(np.average((longitude - weighted_lon) ** 2, weights=weights))

    top_iso = "UNKNOWN"
    top_share = 0.0
    top_three_share = 0.0
    entropy = 0.0
    gini = 0.0

    if not weight_per_iso.empty:
        total_per_iso = weight_per_iso.sum()
        probs = weight_per_iso / total_per_iso if total_per_iso > 0 else weight_per_iso
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        gini = _gini(weight_per_iso.to_numpy(dtype=float))

        top_iso = weight_per_iso.index[0]
        top_share = float(weight_per_iso.iloc[0] / total_per_iso) if total_per_iso > 0 else 0.0
        top_three_share = float(weight_per_iso.iloc[:3].sum() / total_per_iso) if total_per_iso > 0 else 0.0

    return pd.Series(
        {
            "geo_total_addresses": float(total_addresses),
            "geo_num_locations": int(cleaned.shape[0]),
            "geo_num_countries": int(cleaned["country_iso"].nunique()),
            "geo_top_country_iso": top_iso,
            "geo_top_country_share": top_share,
            "geo_top3_country_share": top_three_share,
            "geo_country_entropy": entropy,
            "geo_country_gini": gini,
            "geo_weighted_latitude": weighted_lat,
            "geo_weighted_longitude": weighted_lon,
            "geo_latitude_std": float(np.sqrt(lat_var)),
            "geo_longitude_std": float(np.sqrt(lon_var)),
        }
    )


def _load_geo_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["asn"])
    df["address_count"] = df["address_count"].astype(float)
    aggregated = df.groupby("asn").apply(_compute_geo_features).reset_index()
    return aggregated


def build_dataset(paths: DatasetPaths) -> pd.DataFrame:
    base = _load_peeringdb(paths.peeringdb_all)
    asrank = _load_asrank(paths.asrank_features)
    domains = _load_domain_counts(paths.domain_counts)
    geo = _load_geo_features(paths.geo_aggregated)

    dataset = (
        base.merge(asrank, on="asn", how="left")
        .merge(domains, on="asn", how="left")
        .merge(geo, on="asn", how="left")
    )
    dataset["domain_count"] = dataset["domain_count"].fillna(0.0)
    if "geo_top_country_iso" in dataset.columns:
        dataset["geo_top_country_iso"] = dataset["geo_top_country_iso"].fillna("UNKNOWN")
    return dataset


def _make_preprocessor(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    text_feature: str,
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MaxAbsScaler()),
            ("to_sparse", FunctionTransformer(lambda x: sparse.csr_matrix(x), accept_sparse=True)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: List[Tuple[str, object, List[str] | str]] = [
        ("org_name", TfidfVectorizer(max_features=20000, ngram_range=(1, 2)), text_feature),
    ]

    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.3)
    return preprocessor


def train_classifier(
    dataset: pd.DataFrame,
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> Dict[str, object]:
    feature_columns = [col for col in dataset.columns if col not in {"info_type"}]
    features = dataset[feature_columns].copy()
    targets = dataset["info_type"].astype(str)

    if "asn" in features.columns:
        asn_series = features["asn"].copy()
        features = features.drop(columns=["asn"])
    else:
        asn_series = pd.Series(index=features.index, dtype=int)

    text_feature = "org_name"
    categorical_features = ["geo_top_country_iso"] if "geo_top_country_iso" in features.columns else []
    numeric_features = sorted(
        col
        for col in features.columns
        if col != text_feature and col not in categorical_features
    )

    preprocessor = _make_preprocessor(numeric_features, categorical_features, text_feature)
    classifier = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        verbose=1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        stratify=targets,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    clf_log_loss = log_loss(y_test, y_proba)

    probability_frame = pd.DataFrame(y_proba, index=X_test.index, columns=model.classes_)
    prediction_overview = pd.DataFrame(
        {
            "asn": asn_series.loc[X_test.index].astype("Int64"),
            "org_name": X_test["org_name"],
            "true_label": y_test,
            "predicted_label": y_pred,
            "predicted_confidence": probability_frame.max(axis=1),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    dump(model, output_dir / "asn_type_classifier.joblib")
    prediction_overview.to_csv(output_dir / "holdout_predictions.csv", index=False)

    serializable_report = {}
    for label, stats in report.items():
        if isinstance(stats, dict):
            serializable_report[label] = {metric: float(value) for metric, value in stats.items()}
        else:
            serializable_report[label] = float(stats)

    metrics = {
        "log_loss": clf_log_loss,
        "classification_report": serializable_report,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "test_size": test_size,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    feature_summary = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "text_feature": text_feature,
    }

    with open(output_dir / "feature_summary.json", "w", encoding="utf-8") as fp:
        json.dump(feature_summary, fp, indent=2)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ASN type classifier using PeeringDB data.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("preprocessing/data"),
        help="Base directory containing the input datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/artifacts/asn_type_classifier"),
        help="Directory to store model artifacts and evaluation outputs.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to reserve for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = DatasetPaths(
        peeringdb_all=args.data_root / "peeringdb" / "all_asn.csv",
        asrank_features=args.data_root / "asrank" / "as_rank_df.csv",
        domain_counts=args.data_root / "ipinfo_domains" / "ipinfo_domains.csv",
        geo_aggregated=args.data_root / "geolocation" / "asn_country_stats.csv",
    )

    dataset = build_dataset(paths)
    metrics = train_classifier(
        dataset=dataset,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
