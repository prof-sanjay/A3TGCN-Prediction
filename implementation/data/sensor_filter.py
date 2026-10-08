"""
Data-quality filter: sensors with duplicated (copied) series.

LargeST-GBA contains groups of sensors whose training series are
exactly identical - every value and every NaN position - although
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
) -> list[list[int]]:
    """
    Groups of sensors whose training series are exactly identical,
    including the positions of their missing values.

    Parameters
    ----------
    train_data:
        Training traffic with shape [time, nodes]. NaNs preserved.

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

    buckets: dict[bytes, list[int]] = {}

    for sensor in range(train_data.shape[1]):

        column = train_data[:, sensor]
        missing = np.isnan(column)

        # Exact key: the observed values (missing positions set to a
        # fixed placeholder only inside this key) plus the missing
        # pattern itself. The data array is not modified.
        key = (
            np.where(missing, 0.0, column).astype(np.float64).tobytes()
            + missing.tobytes()
        )

        buckets.setdefault(key, []).append(sensor)

    groups = [
        sorted(members)
        for members in buckets.values()
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
                "identical (values and NaN positions) to another sensor",
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
