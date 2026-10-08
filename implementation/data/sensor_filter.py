"""
Data-quality filter: sensors with duplicated (copied) series.

LargeST-GBA contains groups of sensors whose training series are
identical on every jointly observed timestep (copies that differ
only in a few NaN positions), although
the sensors lie on different freeways tens of kilometres apart.
Such series cannot be independent measurements. They make local
VAR regressions singular and give PEM degenerate covariances
(fake edges between copies).

Ablation decision D9 (option c): every sensor that belongs to such
a group is excluded from E0, E1 and E2 alike.

The groups are identified on the TRAINING split only, so the
validation and test periods do not influence the sensor set.

Traffic values are never modified; sensors are only selected.
"""

import numpy as np


def find_duplicate_sensor_groups(
    train_data: np.ndarray,
    min_overlap: float = 0.5,
) -> list[list[int]]:
    """
    Groups of sensors whose training series are identical on every
    timestep at which BOTH sensors are observed.

    Copies in LargeST-GBA sometimes differ only in a few NaN
    positions, so equality is required on the jointly observed rows
    rather than on the NaN pattern as well. Two sensors are linked
    only if they are jointly observed on at least ``min_overlap`` of
    the training rows; linked sensors are merged into groups
    (connected components).

    Parameters
    ----------
    train_data:
        Training traffic with shape [time, nodes]. NaNs preserved.

    min_overlap:
        Minimum fraction of training rows on which both sensors of
        a pair must be observed.

    Returns
    -------
    groups:
        List of groups (each a sorted list of >= 2 sensor indices),
        ordered by their smallest sensor index.
    """

    if train_data.ndim != 2:
        raise ValueError(
            f"Expected data with shape [time, nodes], "
            f"got {train_data.shape}"
        )

    num_timesteps, num_sensors = train_data.shape

    observed = np.isfinite(train_data)

    # ---------------------------------------------------------
    # Candidate pre-filter: sensors that are identical on every
    # jointly observed row are also identical on the rows where
    # ALL sensors are observed. Bucket on those rows, then verify
    # every candidate pair exactly on its own jointly observed rows.
    # ---------------------------------------------------------

    all_observed = np.all(observed, axis=1)

    buckets: dict[bytes, list[int]] = {}

    for sensor in range(num_sensors):

        key = train_data[all_observed, sensor].tobytes()

        buckets.setdefault(key, []).append(sensor)

    parent = list(range(num_sensors))

    def find(sensor):

        while parent[sensor] != sensor:
            parent[sensor] = parent[parent[sensor]]
            sensor = parent[sensor]

        return sensor

    for members in buckets.values():

        for a_index, a in enumerate(members):

            for b in members[a_index + 1:]:

                both = observed[:, a] & observed[:, b]

                if both.sum() < min_overlap * num_timesteps:
                    continue

                if np.array_equal(
                    train_data[both, a],
                    train_data[both, b],
                ):
                    parent[find(b)] = find(a)

    components: dict[int, list[int]] = {}

    for sensor in range(num_sensors):
        components.setdefault(find(sensor), []).append(sensor)

    groups = [
        sorted(members)
        for members in components.values()
        if len(members) > 1
    ]

    return sorted(groups, key=lambda group: group[0])


def duplicate_filter(
    train_data: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Sensors to KEEP after removing every duplicate-group member.

    Returns
    -------
    keep:
        Ascending int64 array of retained sensor indices.

    report:
        JSON-serialisable description of the filter.
    """

    groups = find_duplicate_sensor_groups(train_data)

    excluded = np.array(
        sorted({sensor for group in groups for sensor in group}),
        dtype=np.int64,
    )

    keep = np.setdiff1d(
        np.arange(train_data.shape[1], dtype=np.int64),
        excluded,
    )

    report = {
        "rule": "exclude every sensor whose training series is "
                "identical to another sensor on all jointly observed "
                "rows (joint observation >= 50% of training rows)",
        "num_sensors_before": int(train_data.shape[1]),
        "num_groups": len(groups),
        "num_excluded": int(len(excluded)),
        "num_sensors_after": int(len(keep)),
        "group_sizes": sorted(
            (len(group) for group in groups),
            reverse=True,
        ),
        "excluded_sensors": excluded.tolist(),
        "groups": groups,
    }

    return keep, report


def select_sensors(
    traffic: np.ndarray,
    adjacency: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Restrict traffic columns and adjacency rows/columns to ``keep``.

    The order of the retained sensors is preserved.
    """

    keep = np.asarray(keep, dtype=np.int64)

    return (
        np.ascontiguousarray(traffic[:, keep]),
        np.ascontiguousarray(adjacency[np.ix_(keep, keep)]),
    )
