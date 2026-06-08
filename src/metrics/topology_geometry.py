# src/metrics/topology_geometry.py
"""Topology / Geometry metrics for Push-T templates.

Fixed version (2026-05-25):
- Proper schema parsing for real reset_templates_v0.json
- Obstacle pose nested under 'pose' key
- blocking_score includes EE→object approach blocking
- passage_width uses pairwise obstacle gaps
- All JSON outputs sanitized (no NaN/Infinity)
- Missing fields recorded as warnings
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np


# Workspace bounds (from MuJoCo env)
WORKSPACE_X_MIN, WORKSPACE_X_MAX = 0.0, 0.70
WORKSPACE_Y_MIN, WORKSPACE_Y_MAX = 0.0, 0.50

# Default sizes
DEFAULT_OBJ_RADIUS = 0.048  # object half-size
DEFAULT_EE_RADIUS = 0.015   # pusher radius


def sanitize_metric_value(x: Any) -> Any:
    """Convert NaN/Inf to None for JSON compatibility."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if np.isnan(x) or np.isinf(x):
            return None
        return float(x)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def sanitize_metrics_dict(d: dict) -> dict:
    """Recursively sanitize all values in a dict."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = sanitize_metrics_dict(v)
        elif isinstance(v, list):
            result[k] = [sanitize_metric_value(x) for x in v]
        else:
            result[k] = sanitize_metric_value(v)
    return result


def extract_template_geometry(template: dict) -> dict:
    """Extract geometry info from a template dict.
    
    Supports real reset_templates_v0.json schema:
    - obstacle pose nested under 'pose' key
    - size_x/size_y as full sizes
    """
    warnings = []
    missing_fields = []
    
    # Template ID
    template_id = template.get('reset_template_id', template.get('template_id', 'unknown'))
    if 'reset_template_id' not in template and 'template_id' not in template:
        missing_fields.append('template_id')
    
    # Source family
    source_family = template.get('layout_family', template.get('family', 'unknown'))
    if 'layout_family' not in template and 'family' not in template:
        missing_fields.append('layout_family')
    
    # Schema version
    schema_version = template.get('schema_version', 'unknown')
    
    # Object pose
    obj_raw = template.get('object_initial_pose', {})
    if not obj_raw:
        missing_fields.append('object_initial_pose')
    obj_xy = np.array([obj_raw.get('x', 0.2), obj_raw.get('y', 0.2)])
    obj_theta = obj_raw.get('theta', 0.0)
    if 'x' not in obj_raw or 'y' not in obj_raw:
        warnings.append('object_initial_pose missing x or y, using defaults')
    
    # Goal pose
    goal_raw = template.get('goal_pose', {})
    if not goal_raw:
        missing_fields.append('goal_pose')
    goal_xy = np.array([goal_raw.get('x', 0.4), goal_raw.get('y', 0.3)])
    goal_theta = goal_raw.get('theta', 0.0)
    if 'x' not in goal_raw or 'y' not in goal_raw:
        warnings.append('goal_pose missing x or y, using defaults')
    
    # EE pose
    ee_raw = template.get('ee_initial_pose', {})
    if not ee_raw:
        missing_fields.append('ee_initial_pose')
    ee_xy = np.array([ee_raw.get('x', 0.1), ee_raw.get('y', 0.2)])
    if 'x' not in ee_raw or 'y' not in ee_raw:
        warnings.append('ee_initial_pose missing x or y, using defaults')
    
    # Object size
    obj_size_x = template.get('object_size_x', DEFAULT_OBJ_RADIUS * 2)
    obj_size_y = template.get('object_size_y', DEFAULT_OBJ_RADIUS * 2)
    obj_radius = max(obj_size_x, obj_size_y) / 2.0  # Use half-size as radius
    
    # Obstacles
    obstacles_raw = template.get('obstacles', [])
    obstacle_poses = []
    obstacle_sizes = []
    obstacle_ids = []
    size_convention_warnings = []
    
    for i, obs in enumerate(obstacles_raw):
        # Obstacle ID
        obs_id = obs.get('obstacle_id', f'obs_{i}')
        obstacle_ids.append(obs_id)
        
        # Obstacle pose (nested under 'pose' key in real templates)
        obs_pose_raw = obs.get('pose', obs)
        obs_x = obs_pose_raw.get('x', None)
        obs_y = obs_pose_raw.get('y', None)
        
        if obs_x is None or obs_y is None:
            warnings.append(f'obstacle {obs_id} missing pose x/y')
            continue
        
        obstacle_poses.append(np.array([obs_x, obs_y]))
        
        # Obstacle size
        # Real templates use size_x/size_y as full sizes
        size_x = obs.get('size_x', None)
        size_y = obs.get('size_y', None)
        
        if size_x is None or size_y is None:
            warnings.append(f'obstacle {obs_id} missing size_x/size_y, using default')
            size_x = 0.06
            size_y = 0.06
        
        # Assume size_x/size_y are full sizes → half-extents for collision
        half_x = size_x / 2.0
        half_y = size_y / 2.0
        obstacle_sizes.append(np.array([half_x, half_y]))
        
        if size_x == size_y:
            size_convention_warnings.append(f'obstacle {obs_id}: size_x == size_y, may be half-size already')
    
    return {
        'template_id': template_id,
        'source_family': source_family,
        'schema_version_detected': schema_version,
        'object_pose': np.array([obj_xy[0], obj_xy[1], obj_theta]),
        'goal_pose': np.array([goal_xy[0], goal_xy[1], goal_theta]),
        'ee_pose': np.array([ee_xy[0], ee_xy[1], 0.0]),
        'obstacle_poses': obstacle_poses,
        'obstacle_sizes': obstacle_sizes,  # half-extents
        'obstacle_ids': obstacle_ids,
        'obstacle_count': len(obstacle_poses),
        'object_shape': template.get('object_shape', 'T'),
        'object_radius': obj_radius,
        'workspace_bounds': np.array([[WORKSPACE_X_MIN, WORKSPACE_Y_MIN],
                                       [WORKSPACE_X_MAX, WORKSPACE_Y_MAX]]),
        'missing_fields': missing_fields,
        'schema_warnings': warnings,
        'size_convention_warnings': size_convention_warnings,
        'size_convention': 'half_extents (size_x/2, size_y/2)',
    }


def compute_basic_geometry_metrics(geometry: dict) -> dict:
    """Compute basic distance/position metrics."""
    obj = geometry['object_pose']
    goal = geometry['goal_pose']
    ee = geometry['ee_pose']
    obs_poses = geometry['obstacle_poses']
    obs_sizes = geometry['obstacle_sizes']
    obj_radius = geometry['object_radius']
    
    # Object-goal distance (center-to-center)
    obj_goal_dist = float(np.linalg.norm(obj[:2] - goal[:2]))
    
    # Goal edge distance (center to boundary)
    gx, gy = goal[0], goal[1]
    goal_edge_dist_raw = float(min(gx - WORKSPACE_X_MIN, WORKSPACE_X_MAX - gx,
                                    gy - WORKSPACE_Y_MIN, WORKSPACE_Y_MAX - gy))
    # Adjusted by object radius
    goal_edge_dist_adjusted = goal_edge_dist_raw - obj_radius
    
    # Object edge distance
    ox, oy = obj[0], obj[1]
    obj_edge_dist = float(min(ox - WORKSPACE_X_MIN, WORKSPACE_X_MAX - ox,
                               oy - WORKSPACE_Y_MIN, WORKSPACE_Y_MAX - oy))
    
    # EE distances
    ee_obj_dist = float(np.linalg.norm(ee[:2] - obj[:2]))
    ee_goal_dist = float(np.linalg.norm(ee[:2] - goal[:2]))
    
    # Obstacle distances (raw signed distance, not clipped)
    min_obj_obs_dist_raw = float('inf')
    min_goal_obs_dist_raw = float('inf')
    min_obj_obs_dist_clipped = float('inf')
    min_goal_obs_dist_clipped = float('inf')
    
    for i, (obs_xy, obs_size) in enumerate(zip(obs_poses, obs_sizes)):
        # Object-obstacle distance (center-to-center minus both radii)
        d_obj = float(np.linalg.norm(obj[:2] - obs_xy)) - obj_radius - np.max(obs_size)
        d_goal = float(np.linalg.norm(goal[:2] - obs_xy)) - obj_radius - np.max(obs_size)
        
        min_obj_obs_dist_raw = min(min_obj_obs_dist_raw, d_obj)
        min_goal_obs_dist_raw = min(min_goal_obs_dist_raw, d_goal)
        min_obj_obs_dist_clipped = min(min_obj_obs_dist_clipped, max(0, d_obj))
        min_goal_obs_dist_clipped = min(min_goal_obs_dist_clipped, max(0, d_goal))
    
    if len(obs_poses) == 0:
        min_obj_obs_dist_raw = float('inf')
        min_goal_obs_dist_raw = float('inf')
        min_obj_obs_dist_clipped = float('inf')
        min_goal_obs_dist_clipped = float('inf')
    
    # Object-goal angle
    diff = goal[:2] - obj[:2]
    obj_goal_angle = float(np.arctan2(diff[1], diff[0]))
    
    # Obstacle density near path
    obs_density = 0.0
    path_len = obj_goal_dist
    if path_len > 0.01:
        path_dir = diff / path_len
        for obs_xy, obs_size in zip(obs_poses, obs_sizes):
            obs_rel = obs_xy - obj[:2]
            proj = float(np.dot(obs_rel, path_dir))
            if -0.1 <= proj <= path_len + 0.1:  # Within path corridor
                perp = obs_rel - proj * path_dir
                perp_dist = float(np.linalg.norm(perp))
                if perp_dist < 0.2:  # Within 20cm of path
                    obs_density += 1.0
    
    return {
        'object_goal_distance': obj_goal_dist,
        'goal_edge_distance_raw': goal_edge_dist_raw,
        'goal_edge_distance_adjusted': max(0, goal_edge_dist_adjusted),
        'object_edge_distance': obj_edge_dist,
        'min_object_obstacle_distance_raw': min_obj_obs_dist_raw,
        'min_object_obstacle_distance_clipped': min_obj_obs_dist_clipped,
        'min_goal_obstacle_distance_raw': min_goal_obs_dist_raw,
        'min_goal_obstacle_distance_clipped': min_goal_obs_dist_clipped,
        'ee_to_object_distance': ee_obj_dist,
        'ee_to_goal_distance': ee_goal_dist,
        'object_goal_angle': obj_goal_angle,
        'obstacle_count': geometry['obstacle_count'],
        'obstacle_density_near_path': obs_density,
    }


def _line_segment_blocked(start: np.ndarray, end: np.ndarray, 
                           obs_poses: list, obs_sizes: list, margin: float = 0.0) -> tuple:
    """Check if line segment from start to end is blocked by obstacles.
    
    Returns: (is_blocked, min_clearance, blocking_obstacle_idx)
    """
    path_vec = end - start
    path_len = float(np.linalg.norm(path_vec))
    if path_len < 1e-6:
        return False, float('inf'), -1
    
    path_dir = path_vec / path_len
    is_blocked = False
    min_clearance = float('inf')
    blocking_idx = -1
    
    for i, (obs_xy, obs_size) in enumerate(zip(obs_poses, obs_sizes)):
        obs_rel = obs_xy - start
        proj = float(np.dot(obs_rel, path_dir))
        
        if -margin <= proj <= path_len + margin:
            perp = obs_rel - proj * path_dir
            perp_dist = float(np.linalg.norm(perp))
            obs_radius = float(np.max(obs_size))
            clearance = perp_dist - obs_radius
            
            if clearance < min_clearance:
                min_clearance = clearance
            if clearance < 0:
                is_blocked = True
                blocking_idx = i
    
    return is_blocked, min_clearance, blocking_idx


def compute_path_topology_metrics(geometry: dict) -> dict:
    """Compute path blocking and passage metrics."""
    obj = geometry['object_pose']
    goal = geometry['goal_pose']
    ee = geometry['ee_pose']
    obs_poses = geometry['obstacle_poses']
    obs_sizes = geometry['obstacle_sizes']
    obj_radius = geometry['object_radius']
    
    # 1. Object→Goal path blocking
    obj_goal_blocked, obj_goal_clearance, _ = _line_segment_blocked(
        obj[:2], goal[:2], obs_poses, obs_sizes, margin=obj_radius)
    
    # 2. EE→Object approach blocking
    ee_obj_blocked, ee_obj_clearance, _ = _line_segment_blocked(
        ee[:2], obj[:2], obs_poses, obs_sizes, margin=DEFAULT_EE_RADIUS)
    
    # 3. Goal region blocking (obstacles near goal)
    goal_region_blocked = False
    goal_region_clearance = float('inf')
    for obs_xy, obs_size in zip(obs_poses, obs_sizes):
        d = float(np.linalg.norm(goal[:2] - obs_xy)) - obj_radius - np.max(obs_size)
        if d < goal_region_clearance:
            goal_region_clearance = d
        if d < 0:
            goal_region_blocked = True
    
    # Blocking scores
    obj_goal_blocking_score = max(0, 1.0 - max(0, obj_goal_clearance) / 0.05) if obj_goal_blocked else 0.0
    ee_obj_blocking_score = max(0, 1.0 - max(0, ee_obj_clearance) / 0.05) if ee_obj_blocked else 0.0
    goal_region_blocking_score = max(0, 1.0 - max(0, goal_region_clearance) / 0.05) if goal_region_blocked else 0.0
    
    # Combined blocking score
    blocking_score = max(obj_goal_blocking_score, ee_obj_blocking_score, goal_region_blocking_score)
    
    # 4. Passage width (pairwise obstacle gaps)
    passage_width = float('inf')
    if len(obs_poses) >= 2:
        # Find minimum gap between any two obstacles
        min_gap = float('inf')
        for i in range(len(obs_poses)):
            for j in range(i + 1, len(obs_poses)):
                center_dist = float(np.linalg.norm(obs_poses[i] - obs_poses[j]))
                size_i = float(np.max(obs_sizes[i]))
                size_j = float(np.max(obs_sizes[j]))
                gap = center_dist - size_i - size_j
                min_gap = min(min_gap, gap)
        passage_width = max(0, min_gap)
    elif len(obs_poses) == 1:
        # Single obstacle: estimate gap as distance to workspace boundary
        obs_xy = obs_poses[0]
        obs_size = float(np.max(obs_sizes[0]))
        # Distance to nearest boundary
        to_boundary = min(obs_xy[0] - WORKSPACE_X_MIN, WORKSPACE_X_MAX - obs_xy[0],
                         obs_xy[1] - WORKSPACE_Y_MIN, WORKSPACE_Y_MAX - obs_xy[1])
        passage_width = max(0, to_boundary - obs_size)
    
    # 5. Edge goal score
    goal_edge_adj = compute_basic_geometry_metrics(geometry)['goal_edge_distance_adjusted']
    max_edge = min(WORKSPACE_X_MAX - WORKSPACE_X_MIN, WORKSPACE_Y_MAX - WORKSPACE_Y_MIN) / 2
    edge_goal_score = max(0, 1.0 - max(0, goal_edge_adj) / max_edge)
    
    # 6. Detour ratio proxy
    obj_goal_dist = float(np.linalg.norm(obj[:2] - goal[:2]))
    if obj_goal_blocked and obj_goal_dist > 0.01:
        # Estimate detour: go around obstacle
        detour_ratio = 1.5  # Placeholder
    else:
        detour_ratio = 1.0
    
    return {
        'object_goal_path_blocked': obj_goal_blocked,
        'object_goal_clearance': obj_goal_clearance,
        'object_goal_blocking_score': obj_goal_blocking_score,
        'ee_object_path_blocked': ee_obj_blocked,
        'ee_object_clearance': ee_obj_clearance,
        'ee_object_blocking_score': ee_obj_blocking_score,
        'goal_region_blocked': goal_region_blocked,
        'goal_region_clearance': goal_region_clearance,
        'goal_region_blocking_score': goal_region_blocking_score,
        'blocking_score': blocking_score,
        'passage_width_estimate': passage_width,
        'edge_goal_score': edge_goal_score,
        'detour_ratio_proxy': detour_ratio,
    }


def compute_contact_access_metrics(geometry: dict) -> dict:
    """Compute pusher approach and contact metrics."""
    obj = geometry['object_pose']
    goal = geometry['goal_pose']
    ee = geometry['ee_pose']
    obs_poses = geometry['obstacle_poses']
    obs_sizes = geometry['obstacle_sizes']
    obj_radius = geometry['object_radius']
    
    obj_xy = obj[:2]
    goal_xy = goal[:2]
    ee_xy = ee[:2]
    
    # Goal direction (required push direction)
    goal_dir = goal_xy - obj_xy
    goal_dir_norm = float(np.linalg.norm(goal_dir))
    if goal_dir_norm > 1e-6:
        goal_dir = goal_dir / goal_dir_norm
    else:
        goal_dir = np.array([1.0, 0.0])
    
    # Check 4 contact sides
    contact_dirs = {
        'left': np.array([-1.0, 0.0]),
        'right': np.array([1.0, 0.0]),
        'bottom': np.array([0.0, -1.0]),
        'top': np.array([0.0, 1.0]),
    }
    
    reachable_sides = []
    side_clearances = {}
    
    for side_name, dir_vec in contact_dirs.items():
        # Contact point on object boundary
        contact_pt = obj_xy + dir_vec * obj_radius
        
        # Check workspace bounds
        if not (WORKSPACE_X_MIN <= contact_pt[0] <= WORKSPACE_X_MAX and
                WORKSPACE_Y_MIN <= contact_pt[1] <= WORKSPACE_Y_MAX):
            side_clearances[side_name] = -1.0  # Out of bounds
            continue
        
        # Check clearance from obstacles
        min_clearance = float('inf')
        for obs_xy, obs_size in zip(obs_poses, obs_sizes):
            d = float(np.linalg.norm(contact_pt - obs_xy)) - np.max(obs_size) - DEFAULT_EE_RADIUS
            min_clearance = min(min_clearance, d)
        
        side_clearances[side_name] = min_clearance
        if min_clearance > 0:
            reachable_sides.append(side_name)
    
    # Best contact side (closest to goal direction)
    best_side = None
    best_dot = -2.0
    for side_name in reachable_sides:
        dot = float(np.dot(contact_dirs[side_name], goal_dir))
        if dot > best_dot:
            best_dot = dot
            best_side = side_name
    
    # Best contact side clearance
    best_clearance = side_clearances.get(best_side, 0.0) if best_side else 0.0
    
    # EE→Object path analysis
    ee_obj_blocked, ee_obj_clearance, _ = _line_segment_blocked(
        ee_xy, obj_xy, obs_poses, obs_sizes, margin=DEFAULT_EE_RADIUS)
    
    # Alignment: how well does EE→object direction align with required push direction
    ee_obj_vec = obj_xy - ee_xy
    ee_obj_dist = float(np.linalg.norm(ee_obj_vec))
    if ee_obj_dist > 1e-6:
        ee_obj_dir = ee_obj_vec / ee_obj_dist
        alignment = float(np.dot(ee_obj_dir, goal_dir))
    else:
        alignment = 1.0  # Already at object
    
    # Approach feasibility score (composite)
    dist_score = max(0, 1.0 - ee_obj_dist / 0.3)  # 30cm max
    clearance_score = min(1.0, max(0, ee_obj_clearance) / 0.05)  # 5cm = 1.0
    side_score = len(reachable_sides) / 4.0
    alignment_score = max(0, alignment)  # 0 to 1
    blocked_penalty = 0.5 if ee_obj_blocked else 0.0
    
    approach_score = (dist_score * 0.2 + clearance_score * 0.3 + side_score * 0.2 + 
                     alignment_score * 0.3 - blocked_penalty)
    approach_score = max(0, min(1, approach_score))
    
    return {
        'reachable_contact_sides': len(reachable_sides),
        'reachable_contact_side_names': reachable_sides,
        'best_contact_side': best_side,
        'best_contact_side_clearance': best_clearance,
        'side_clearances': side_clearances,
        'ee_object_path_blocked': ee_obj_blocked,
        'ee_object_path_clearance': ee_obj_clearance,
        'ee_object_goal_alignment': alignment,
        'approach_feasibility_score': approach_score,
    }


def classify_template_topology(geometry: dict, basic_metrics: dict,
                                path_metrics: dict, contact_metrics: dict) -> dict:
    """Classify template topology and difficulty."""
    # Difficulty score (0=easy, 1=hard)
    difficulty = 0.0
    
    # Object-goal distance factor
    og_dist = basic_metrics['object_goal_distance']
    difficulty += min(0.2, og_dist / 0.5 * 0.2)
    
    # Blocking factor (use max of all blocking scores)
    blocking = path_metrics['blocking_score']
    difficulty += blocking * 0.25
    
    # Passage narrowness
    pw = path_metrics['passage_width_estimate']
    if pw is not None and pw < float('inf'):
        difficulty += max(0, 0.2 * (1.0 - pw / 0.1))
    
    # Edge goal factor
    difficulty += path_metrics['edge_goal_score'] * 0.15
    
    # Approach difficulty
    difficulty += (1.0 - contact_metrics['approach_feasibility_score']) * 0.2
    
    difficulty = min(1.0, max(0.0, difficulty))
    
    # Difficulty level
    if difficulty < 0.3:
        difficulty_level = 'easy'
    elif difficulty < 0.6:
        difficulty_level = 'medium'
    else:
        difficulty_level = 'hard'
    
    # Dominant challenge
    challenges = []
    if path_metrics['ee_object_blocking_score'] > 0.3:
        challenges.append('ee_approach_blocked')
    if path_metrics['object_goal_blocking_score'] > 0.3:
        challenges.append('object_goal_blocked')
    if path_metrics['goal_region_blocking_score'] > 0.3:
        challenges.append('goal_region_blocked')
    if path_metrics['edge_goal_score'] > 0.5:
        challenges.append('edge_goal')
    if pw is not None and pw < 0.05:
        challenges.append('narrow_passage')
    if contact_metrics['approach_feasibility_score'] < 0.3:
        challenges.append('approach_difficult')
    if not challenges:
        challenges.append('none')
    
    # Topology family prediction
    if path_metrics['object_goal_blocking_score'] > 0.3 or path_metrics['ee_object_blocking_score'] > 0.3:
        topology_pred = 'blocking'
    elif path_metrics['passage_width_estimate'] is not None and path_metrics['passage_width_estimate'] < 0.08:
        topology_pred = 'narrow_passage'
    elif path_metrics['edge_goal_score'] > 0.5:
        topology_pred = 'edge_goal'
    elif geometry['obstacle_count'] == 0:
        topology_pred = 'open_space'
    else:
        topology_pred = 'non_blocking'
    
    # Warnings
    warnings = []
    if basic_metrics['object_goal_distance'] < 0.05:
        warnings.append('object very close to goal')
    if basic_metrics['goal_edge_distance_adjusted'] < 0:
        warnings.append('goal too close to boundary (adjusted < 0)')
    if basic_metrics['min_object_obstacle_distance_raw'] < 0:
        warnings.append('object overlaps with obstacle')
    if contact_metrics['reachable_contact_sides'] == 0:
        warnings.append('no reachable contact sides')
    
    return {
        'difficulty_score': float(difficulty),
        'difficulty_level': difficulty_level,
        'dominant_geometric_challenge': challenges[0],
        'all_challenges': challenges,
        'topology_family_pred': topology_pred,
        'warnings': warnings,
    }


def validate_template_geometry(geometry: dict, basic_metrics: dict) -> dict:
    """Validate template geometry."""
    invalid_reasons = []
    warnings = []
    
    # Check workspace bounds
    obj = geometry['object_pose']
    goal = geometry['goal_pose']
    
    if not (WORKSPACE_X_MIN <= obj[0] <= WORKSPACE_X_MAX and
            WORKSPACE_Y_MIN <= obj[1] <= WORKSPACE_Y_MAX):
        invalid_reasons.append(f'object out of bounds: ({obj[0]:.3f}, {obj[1]:.3f})')
    
    if not (WORKSPACE_X_MIN <= goal[0] <= WORKSPACE_X_MAX and
            WORKSPACE_Y_MIN <= goal[1] <= WORKSPACE_Y_MAX):
        invalid_reasons.append(f'goal out of bounds: ({goal[0]:.3f}, {goal[1]:.3f})')
    
    # Check obstacles in bounds
    for i, obs_xy in enumerate(geometry['obstacle_poses']):
        if not (WORKSPACE_X_MIN <= obs_xy[0] <= WORKSPACE_X_MAX and
                WORKSPACE_Y_MIN <= obs_xy[1] <= WORKSPACE_Y_MAX):
            invalid_reasons.append(f'obstacle {i} out of bounds')
    
    # Check overlaps (use raw signed distance)
    if basic_metrics['min_object_obstacle_distance_raw'] < 0:
        invalid_reasons.append(f'object overlaps with obstacle (raw distance: {basic_metrics["min_object_obstacle_distance_raw"]:.4f})')
    
    if basic_metrics['min_goal_obstacle_distance_raw'] < 0:
        warnings.append(f'goal overlaps with obstacle (raw distance: {basic_metrics["min_goal_obstacle_distance_raw"]:.4f})')
    
    # Check missing required fields
    if geometry.get('missing_fields'):
        for field in geometry['missing_fields']:
            if field in ['object_initial_pose', 'goal_pose']:
                invalid_reasons.append(f'missing required field: {field}')
            else:
                warnings.append(f'missing field: {field}')
    
    return {
        'is_valid': len(invalid_reasons) == 0,
        'invalid_reasons': invalid_reasons,
        'warnings': warnings,
    }


def compute_all_metrics(template: dict) -> dict:
    """Compute all topology/geometry metrics for a template."""
    geometry = extract_template_geometry(template)
    basic = compute_basic_geometry_metrics(geometry)
    path = compute_path_topology_metrics(geometry)
    contact = compute_contact_access_metrics(geometry)
    classification = classify_template_topology(geometry, basic, path, contact)
    validation = validate_template_geometry(geometry, basic)
    
    # Sanitize all metrics for JSON output
    result = {
        'template_id': geometry['template_id'],
        'source_family': geometry['source_family'],
        'schema_version_detected': geometry['schema_version_detected'],
        'geometry': {
            'object_pose': geometry['object_pose'].tolist(),
            'goal_pose': geometry['goal_pose'].tolist(),
            'ee_pose': geometry['ee_pose'].tolist(),
            'obstacle_count': geometry['obstacle_count'],
            'object_shape': geometry['object_shape'],
            'object_radius': geometry['object_radius'],
            'size_convention': geometry['size_convention'],
        },
        'basic_metrics': basic,
        'path_metrics': path,
        'contact_metrics': {k: v for k, v in contact.items()
                           if k not in ['reachable_contact_side_names', 'side_clearances']},
        'classification': classification,
        'validation': validation,
        'missing_fields': geometry['missing_fields'],
        'schema_warnings': geometry['schema_warnings'],
        'size_convention_warnings': geometry['size_convention_warnings'],
    }
    
    return sanitize_metrics_dict(result)
