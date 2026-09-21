"""Pure identity checks for one immutable ReKep perception snapshot."""


def validate_snapshot_layout(keypoints, instances):
    """Return ``(valid, reason)`` for K-to-rigid-instance identity."""
    points = list(keypoints)
    objects = list(instances)
    ids = [int(item.id) for item in points]
    if ids != list(range(len(ids))):
        return False, "keypoint_ids_must_be_contiguous_K0_to_Kn"
    if any(item.name != "K{}".format(index)
           for index, item in enumerate(points)):
        return False, "keypoint_names_must_match_visual_K_indices"
    groups = [int(item.rigid_group_id) for item in objects]
    if any(value <= 0 for value in groups) or len(groups) != len(set(groups)):
        return False, "instances_must_have_unique_positive_rigid_groups"
    known = set(groups)
    if any(int(item.rigid_group_id) not in known for item in points):
        return False, "every_keypoint_must_bind_one_rigid_instance"
    return True, "immutable_snapshot_locked"


def groups_in_seed_bounds(instances, lower, upper):
    """Select initial object regions by their measured surface representative."""
    return {int(item.rigid_group_id) for item in instances
            if all(lo <= value <= hi for lo, value, hi in zip(
                lower, (item.surface_medoid.x, item.surface_medoid.y,
                        item.surface_medoid.z), upper))}
