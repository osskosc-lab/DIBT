from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EstimationConfig
from .estimators import (
    InterventionalInput,
    ObservationalInput,
    make_estimator_input,
    score_cmi,
    score_do,
)
from .r2a_config import ClusteringConfig


@dataclass(frozen=True)
class BlindInput:
    transition_inputs: np.ndarray
    next_states: np.ndarray
    intervention_index: np.ndarray
    intervention_value: np.ndarray


@dataclass(frozen=True)
class BlindEstimate:
    cluster_labels: np.ndarray
    role_labels: np.ndarray
    edge_scores_cmi: np.ndarray
    edge_scores_do: np.ndarray
    boundary_relations: np.ndarray
    selected_k: int
    silhouette: float


def _zscore_columns(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (features - mean) / std


def _pairwise_distances(features: np.ndarray) -> np.ndarray:
    differences = features[:, None, :] - features[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=2))


def _average_distance(
    distances: np.ndarray, left: tuple[int, ...], right: tuple[int, ...]
) -> float:
    return float(np.mean(distances[np.ix_(left, right)]))


def _agglomerative_levels(distances: np.ndarray) -> dict[int, np.ndarray]:
    clusters = [tuple([index]) for index in range(len(distances))]
    levels: dict[int, np.ndarray] = {}
    while len(clusters) >= 2:
        ordered = sorted(clusters, key=lambda cluster: tuple(cluster))
        labels = np.empty(len(distances), dtype=int)
        for label, cluster in enumerate(ordered):
            labels[list(cluster)] = label
        levels[len(clusters)] = labels
        if len(clusters) == 2:
            break
        candidates = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                candidates.append(
                    (
                        _average_distance(
                            distances, clusters[left], clusters[right]
                        ),
                        tuple(sorted(clusters[left] + clusters[right])),
                        left,
                        right,
                    )
                )
        _, merged, left, right = min(candidates, key=lambda item: (item[0], item[1]))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left, right}
        ] + [merged]
    return levels


def _silhouette(distances: np.ndarray, labels: np.ndarray) -> float:
    values = []
    for index in range(len(labels)):
        same = np.where(labels == labels[index])[0]
        same = same[same != index]
        if len(same) == 0:
            values.append(0.0)
            continue
        within = float(np.mean(distances[index, same]))
        other_means = [
            float(np.mean(distances[index, labels == other]))
            for other in np.unique(labels)
            if other != labels[index]
        ]
        nearest = min(other_means)
        denominator = max(within, nearest)
        values.append(0.0 if denominator == 0 else (nearest - within) / denominator)
    return float(np.mean(values))


def cluster_profiles(
    features: np.ndarray, cfg: ClusteringConfig
) -> tuple[np.ndarray, int, float]:
    distances = _pairwise_distances(_zscore_columns(features))
    levels = _agglomerative_levels(distances)
    maximum = min(cfg.max_clusters, len(features) - 1)
    choices = []
    for k in range(cfg.min_clusters, maximum + 1):
        labels = levels[k]
        choices.append((_silhouette(distances, labels), k, labels))
    best_score, best_k, best_labels = choices[0]
    for score, k, labels in choices[1:]:
        if score > best_score + cfg.exact_tie_tolerance:
            best_score, best_k, best_labels = score, k, labels
    return best_labels.copy(), best_k, best_score


def _role_labels(cluster_labels: np.ndarray, do_scores: np.ndarray) -> np.ndarray:
    clusters = np.unique(cluster_labels)
    cross_strength = {}
    for cluster in clusters:
        members = np.where(cluster_labels == cluster)[0]
        others = np.where(cluster_labels != cluster)[0]
        cross_strength[cluster] = float(
            np.sum(do_scores[np.ix_(others, members)])
            + np.sum(do_scores[np.ix_(members, others)])
        )
    boundary_cluster = max(
        clusters, key=lambda cluster: (cross_strength[cluster], -int(cluster))
    )
    remaining = [cluster for cluster in clusters if cluster != boundary_cluster]
    role_by_cluster = {int(boundary_cluster): 1}
    if remaining:
        boundary_nodes = np.where(cluster_labels == boundary_cluster)[0]
        net_flow = {}
        for cluster in remaining:
            nodes = np.where(cluster_labels == cluster)[0]
            outgoing = np.sum(do_scores[np.ix_(boundary_nodes, nodes)])
            incoming = np.sum(do_scores[np.ix_(nodes, boundary_nodes)])
            net_flow[cluster] = float(outgoing - incoming)
        external_cluster = max(
            remaining, key=lambda cluster: (net_flow[cluster], -int(cluster))
        )
        role_by_cluster[int(external_cluster)] = 0
        internal_candidates = [
            cluster for cluster in remaining if cluster != external_cluster
        ]
        if internal_candidates:
            internal_cluster = min(
                internal_candidates,
                key=lambda cluster: (net_flow[cluster], int(cluster)),
            )
            role_by_cluster[int(internal_cluster)] = 2
            for cluster in internal_candidates:
                if cluster != internal_cluster:
                    role_by_cluster[int(cluster)] = 3
    return np.asarray(
        [role_by_cluster.get(int(cluster), 3) for cluster in cluster_labels],
        dtype=int,
    )


def _boundary_relations(role_labels: np.ndarray) -> np.ndarray:
    boundary = role_labels == 1
    relation = boundary[:, None] ^ boundary[None, :]
    np.fill_diagonal(relation, False)
    return relation


def _scores(data: BlindInput, estimation_cfg: EstimationConfig):
    observational = ObservationalInput(data.transition_inputs, data.next_states)
    interventional = InterventionalInput(
        data.intervention_index, data.intervention_value, data.next_states
    )
    return (
        score_cmi(observational, estimation_cfg),
        score_do(interventional, estimation_cfg),
    )


def infer_integrated_partition(
    data: BlindInput,
    estimation_cfg: EstimationConfig,
    clustering_cfg: ClusteringConfig,
) -> BlindEstimate:
    cmi_scores, do_scores = _scores(data, estimation_cfg)
    observational_profile = np.concatenate([cmi_scores.T, cmi_scores], axis=1)
    intervention_profile = do_scores.T
    observational_profile = _zscore_columns(observational_profile) / np.sqrt(
        observational_profile.shape[1]
    )
    intervention_profile = _zscore_columns(intervention_profile) / np.sqrt(
        intervention_profile.shape[1]
    )
    features = np.concatenate(
        [observational_profile, intervention_profile], axis=1
    )
    labels, selected_k, silhouette = cluster_profiles(features, clustering_cfg)
    roles = _role_labels(labels, do_scores)
    return BlindEstimate(
        cluster_labels=labels,
        role_labels=roles,
        edge_scores_cmi=cmi_scores,
        edge_scores_do=do_scores,
        boundary_relations=_boundary_relations(roles),
        selected_k=selected_k,
        silhouette=silhouette,
    )


def infer_correlation_baseline(
    data: BlindInput,
    estimation_cfg: EstimationConfig,
    clustering_cfg: ClusteringConfig,
) -> BlindEstimate:
    cmi_scores, do_scores = _scores(data, estimation_cfg)
    correlation = np.nan_to_num(
        np.abs(np.corrcoef(data.transition_inputs, rowvar=False))
    )
    labels, selected_k, silhouette = cluster_profiles(
        correlation, clustering_cfg
    )
    roles = _role_labels(labels, do_scores)
    return BlindEstimate(
        labels,
        roles,
        cmi_scores,
        do_scores,
        _boundary_relations(roles),
        selected_k,
        silhouette,
    )


def infer_do_profile_baseline(
    data: BlindInput,
    estimation_cfg: EstimationConfig,
    clustering_cfg: ClusteringConfig,
) -> BlindEstimate:
    cmi_scores, do_scores = _scores(data, estimation_cfg)
    labels, selected_k, silhouette = cluster_profiles(
        do_scores.T, clustering_cfg
    )
    roles = _role_labels(labels, do_scores)
    return BlindEstimate(
        labels,
        roles,
        cmi_scores,
        do_scores,
        _boundary_relations(roles),
        selected_k,
        silhouette,
    )


def infer_random_baseline(
    data: BlindInput,
    estimation_cfg: EstimationConfig,
    clustering_cfg: ClusteringConfig,
    seed: int,
) -> BlindEstimate:
    cmi_scores, do_scores = _scores(data, estimation_cfg)
    rng = np.random.default_rng(seed)
    maximum = min(clustering_cfg.max_clusters, len(data.transition_inputs[0]) - 1)
    k = int(rng.integers(clustering_cfg.min_clusters, maximum + 1))
    labels = np.concatenate(
        [np.arange(k), rng.integers(0, k, size=len(data.transition_inputs[0]) - k)]
    )
    labels = labels[rng.permutation(len(labels))]
    roles = _role_labels(labels, do_scores)
    return BlindEstimate(
        labels,
        roles,
        cmi_scores,
        do_scores,
        _boundary_relations(roles),
        k,
        0.0,
    )


def estimator_input_for_edges(data: BlindInput):
    return make_estimator_input(
        data.transition_inputs,
        data.next_states,
        data.intervention_index,
        data.intervention_value,
        split_name="TRAIN",
    )
