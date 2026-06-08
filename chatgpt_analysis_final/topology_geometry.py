# src/metrics/topology_geometry.py
"""Topology / Geometry metrics for Push-T templates.

Pure Python / numpy implementation. No MuJoCo dependency.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


# Workspace bounds (from MuJoCo env)
WORKSPACE_X_MIN, WORKSPACE_X_MAX = 0.0, 0.70
WORKSPACE_Y_MIN, WORKSPACE_Y_MAX = 0.0, 0.50
WORKSPACE_CENTER = np.array([(WORKSPACE_X_MIN + WORKSPACE_X_MAX) / 2,
                              (WORKSPACE_Y_MIN + WORKSPACE_Y_MAX) / 2])
WORKSPACE_SIZE = np.array([WORKSPACE_X_MAX - WORKSPACE_X_MIN,
                           WORKSPACE_Y_MAX - WORKSPACE_Y_MIN])

# Default object/goal sizes
DEFAULT_OBJ_SIZE = 0.048  # half-size
DEFAULT_EE_SIZE = 0.015


def extract_template_geometry(template: dict) -> dict:
    """Extract geometry info from a template dict."""
    obj = template.get('object_initial_pose', {})
    goal = template.get('goal_pose', {})
    ee = template.get('ee_initial_pose', {})
    obstacles = template.get('obstacles', [])
    
    obj_xy = np.array([obj.get('x', 0.2), obj.get('y', 0.2)])
    obj_theta = obj.get('theta', 0.0)
    goal_xy = np.array([goal.get('x', 0.4), goal.get('y', 0.3)])
    goal_theta = goal.get('theta', 0.0)
    ee_xy = np.array([ee.get('x', 0.1), ee.get('y', 0.2)])
    
    obstacle_poses = []
    obstacle_sizes = []
    for obs in obstacles:
        obs_xy = np.array([obs.get('x', 0), obs.get('y', 0)])
        obs_size = np.array([obs.get('size_x', 0.05), obs.get('size_y', 0.05)])
        obstacle_poses.append(obs_xy)
        obstacle_sizes.append(obs_size)
    
    return {
        'object_pose': np.array([obj_xy[0], obj_xy[1], obj_theta]),
        'goal_pose': np.array([goal_xy[0], goal_xy[1], goal_theta]),
        'ee_pose': np.array([ee_xy[0], ee_xy[1], 0.0]),
        'obstacle_poses': obstacle_poses,
        'obstacle_sizes': obstacle_sizes,
        'obstacle_count': len(obstacles),
        'object_shape': template.get('object_shape', 'T'),
        'object_size': np.array([template.get('object_size_x', DEFAULT_OBJ_SIZE),
                                  template.get('object_size_y', DEFAULT_OBJ_SIZE)]),
        'workspace_bounds': np.array([[WORKSPACE_X_MIN, WORKSPACE_Y_MIN],
                                       [WORKSPACE_X_MAX, WORKSPACE_Y_MAX]]),
    }


def compute_basic_geometry_metrics(geometry: dict) -> dict:
    """Compute basic distance/position metrics."""
    obj = geometry['object_pose']
    goal = geometry['goal_pose']
    ee = geometry['ee_pose']
    obs_poses = geometry['obstacle_poses']
    obs_sizes = geometry['obstacle_sizes']
    
    # Object-goal distance
    obj_goal_dist = float(np.linalg.norm(obj[:2] - goal[:2]))
    
    # Goal edge distance
    gx, gy = goal[0], goal[1]
    goal_edge_dist = float(min(gx - WORKSPACE_X_MIN, WORKSPACE_X_MAX - gx,
                                gy - WORKSPACE_Y_MIN, WORKSPACE_Y_MAX - gy))
    
    # Object edge distance
    ox, oy = obj[0], obj[1]
    obj_edge_dist = float(min(ox - WORKSPACE_X_MIN, WORKSPACE_X_MAX - ox,
                               oy - WORKSPACE_Y_MIN, WORKSPACE_Y_MAX - oy))
    
    # Obstacle distances
    min_obj_obs_dist = float('inf')
    min_goal_obs_dist = float('inf')
    for i, (obs_xy, obs_size) in enumerate(zip(obs_poses, obs_sizes)):
        # Approximate distance (center-to-center minus sizes)
        d_obj = float(np.linalg.norm(obj[:2] - obs_xy)) - np.max(obs_size)
        d_goal = float(np.linalg.norm(goal[:2] - obs_xy)) - np.max(obs_size)
        min_obj_obs_dist = min(min_obj_obs_dist, max(0, d_obj))
        min_goal_obs_dist = min(min_goal_obs_dist, max(0, d_goal))
    
    if len(obs_poses) == 0:
        min_obj_obs_dist = float('inf')
        min_goal_obs_dist = float('inf')
    
    # EE distances
    ee_obj_dist = float(np.linalg.norm(ee[:2] - obj[:2]))
    ee_goal_dist = float(np.linalg.norm(ee[:2] - goal[:2]))
    
    # Object-goal angle
    diff = goal[:2] - obj[:2]
    obj_goal_angle = float(np.arctan2(diff[1], diff[0]))
    
    return {
        'object_goal_distance': obj_goal_dist,
        'goal_edge_distance': goal_edge_dist,
        'object_edge_distance': obj_edge_dist,
        'min_object_obstacle_distance': min_obj_obs_dist,
        'min_goal_obstacle_distance': min_goal_obs_dist,
        'ee_to_object_distance': ee_obj_dist,
        'ee_to_goal_distance': ee_goal_dist,
        'object_goal_angle': obj_goal_angle,
        'obstacle_count': geometry['obstacle_count'],
    }


def compute_path_topology_metrics(geometry: dict) -> dict:
    """Compute path blocking and passage metrics."""
    obj = geometry['object_pose']
    goal = geometry['goal_pose']
    obs_poses = geometry['obstacle_poses']
    obs_sizes = geometry['obstacle_sizes']
    
    obj_xy = obj[:2]
    goal_xy = goal[:2]
    path_vec = goal_xy - obj_xy
    path_len = float(np.linalg.norm(path_vec))
    
    if path_len < 1e-6:
        return {
            'direct_path_blocked': False,
            'obstacle_between_object_goal': False,
            'blocking_score': 0.0,
            'passage_width_estimate': float('inf'),
            'edge_goal_score': 0.0,
        }
    
    path_dir = path_vec / path_len
    
    # Check if obstacles block the direct path
    direct_blocked = False
    obs_between = False
    blocking_score = 0.0
    min_clearance = float('inf')
    
    for i, (obs_xy, obs_size) in enumerate(zip(obs_poses, obs_sizes)):
        # Project obstacle center onto path line
        obs_rel = obs_xy - obj_xy
        proj = float(np.dot(obs_rel, path_dir))
        
        # Check if projection is between obj and goal
        if 0 <= proj <= path_len:
            # Perpendicular distance from obstacle to path
            perp = obs_rel - proj * path_dir
            perp_dist = float(np.linalg.norm(perp))
            
            # Obstacle effective radius
            obs_radius = float(np.max(obs_size))
            
            if perp_dist < obs_radius:
                direct_blocked = True
                obs_between = True
                blocking_score = max(blocking_score, 1.0 - perp_dist / obs_radius)
            
            min_clearance = min(min_clearance, max(0, perp_dist - obs_radius))
    
    # Passage width estimate (simplified)
    if len(obs_poses) >= 2:
        # Find the two closest obstacles to the path
        obs_dists = []
        for i, (obs_xy, obs_size) in enumerate(zip(obs_poses, obs_sizes)):
            obs_rel = obs_xy - obj_xy
            proj = float(np.dot(obs_rel, path_dir))
            if 0 <= proj <= path_len:
                perp = obs_rel - proj * path_dir
                perp_dist = float(np.linalg.norm(perp))
                obs_dists.append((perp_dist, float(np.max(obs_size))))
        
        if len(obs_dists) >= 2:
            obs_dists.sort()
            passage_width = obs_dists[1][0] - obs_dists[0][1] - obs_dists[1][1]
            passage_width = max(0, passage_width)
        else:
            passage_width = float('inf')
    else:
        passage_width = float('inf')
    
    # Edge goal score
    goal_edge_dist = compute_basic_geometry_metrics(geometry)['goal_edge_distance']
    max_edge = min(WORKSPACE_X_MAX - WORKSPACE_X_MIN, WORKSPACE_Y_MAX - WORKSPACE_Y_MIN) / 2
    edge_goal_score = max(0, 1.0 - goal_edge_dist / max_edge)
    
    return {
        'direct_path_blocked': direct_blocked,
        'obstacle_between_object_goal': obs_between,
        'blocking_score': float(blocking_score),
        'passage_width_estimate': float(passage_width),
        'min_clearance_to_obstacle': float(min_clearance) if min_clearance < float('inf') else float('inf'),
        'edge_goal_score': float(edge_goal_score),
    }


def compute_contact_access_metrics(geometry: dict) -> dict:
    """Compute pusher approach and contact metrics."""
    obj = geometry['object_pose']
    goal = geometry['goal_pose']
    ee = geometry['ee_pose']
    obs_poses = geometry['obstacle_poses']
    obs_sizes = geometry['obstacle_sizes']
    obj_size = geometry['object_size']
    
    obj_xy = obj[:2]
    goal_xy = goal[:2]
    ee_xy = ee[:2]
    
    # Check 4 contact sides (left, right, top, bottom)
    contact_dirs = {
        'left': np.array([-1.0, 0.0]),
        'right': np.array([1.0, 0.0]),
        'bottom': np.array([0.0, -1.0]),
        'top': np.array([0.0, 1.0]),
    }
    
    reachable_sides = []
    for side_name, dir_vec in contact_dirs.items():
        # Contact point on object boundary
        contact_pt = obj_xy + dir_vec * np.max(obj_size)
        
        # Check if contact point is in workspace
        if not (WORKSPACE_X_MIN <= contact_pt[0] <= WORKSPACE_X_MAX and
                WORKSPACE_Y_MIN <= contact_pt[1] <= WORKSPACE_Y_MAX):
            continue
        
        # Check clearance from obstacles
        clearance_ok = True
        for obs_xy, obs_size in zip(obs_poses, obs_sizes):
            d = float(np.linalg.norm(contact_pt - obs_xy)) - np.max(obs_size)
            if d < DEFAULT_EE_SIZE * 2:
                clearance_ok = False
                break
        
        if clearance_ok:
            reachable_sides.append(side_name)
    
    # Best contact side (closest to goal direction)
    goal_dir = goal_xy - obj_xy
    goal_dir_norm = float(np.linalg.norm(goal_dir))
    if goal_dir_norm > 1e-6:
        goal_dir = goal_dir / goal_dir_norm
    
    best_side = None
    best_dot = -2.0
    for side_name in reachable_sides:
        dot = float(np.dot(contact_dirs[side_name], goal_dir))
        if dot > best_dot:
            best_dot = dot
            best_side = side_name
    
    # Approach feasibility
    ee_obj_dist = float(np.linalg.norm(ee_xy - obj_xy))
    approach_clearance = float('inf')
    for obs_xy, obs_size in zip(obs_poses, obs_sizes):
        # Check clearance along ee->obj path
        obs_rel = obs_xy - ee_xy
        path_vec = obj_xy - ee_xy
        path_len = float(np.linalg.norm(path_vec))
        if path_len < 1e-6:
            continue
        path_dir = path_vec / path_len
        proj = float(np.dot(obs_rel, path_dir))
        if 0 <= proj <= path_len:
            perp = obs_rel - proj * path_dir
            perp_dist = float(np.linalg.norm(perp)) - np.max(obs_size)
            approach_clearance = min(approach_clearance, max(0, perp_dist))
    
    if len(obs_poses) == 0:
        approach_clearance = float('inf')
    
    # Approach feasibility score
    if ee_obj_dist < 1e-6:
        approach_score = 1.0
    else:
        dist_score = max(0, 1.0 - ee_obj_dist / 0.3)  # Normalize by 30cm
        clearance_score = min(1.0, approach_clearance / 0.05)  # 5cm clearance = 1.0
        side_score = len(reachable_sides) / 4.0
        approach_score = (dist_score + clearance_score + side_score) / 3.0
    
    # EE-object-goal alignment
    if ee_obj_dist > 1e-6 and goal_dir_norm > 1e-6:
        ee_obj_dir = (obj_xy - ee_xy) / ee_obj_dist
        alignment = float(np.dot(ee_obj_dir, goal_dir))
    else:
        alignment = 0.0
    
    return {
        'reachable_contact_sides': len(reachable_sides),
        'reachable_contact_side_names': reachable_sides,
        'best_contact_side': best_side,
        'approach_clearance': float(approach_clearance) if approach_clearance < float('inf') else float('inf'),
        'approach_feasibility_score': float(approach_score),
        'ee_object_goal_alignment': float(alignment),
    }


def classify_template_topology(geometry: dict, basic_metrics: dict, 
                                path_metrics: dict, contact_metrics: dict) -> dict:
    """Classify template topology and difficulty."""
    # Difficulty score (0=easy, 1=hard)
    difficulty = 0.0
    
    # Object-goal distance factor
    og_dist = basic_metrics['object_goal_distance']
    difficulty += min(0.3, og_dist / 0.5 * 0.3)  # 50cm = max distance factor
    
    # Blocking factor
    difficulty += path_metrics['blocking_score'] * 0.3
    
    # Edge goal factor
    difficulty += path_metrics['edge_goal_score'] * 0.2
    
    # Contact access factor
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
    if path_metrics['blocking_score'] > 0.3:
        challenges.append('blocking')
    if path_metrics['edge_goal_score'] > 0.5:
        challenges.append('edge_goal')
    if contact_metrics['approach_feasibility_score'] < 0.3:
        challenges.append('approach_difficult')
    if basic_metrics['obstacle_count'] > 2:
        challenges.append('many_obstacles')
    if not challenges:
        challenges.append('none')
    
    # Topology family prediction
    if path_metrics['blocking_score'] > 0.5:
        topology_pred = 'blocking'
    elif path_metrics['passage_width_estimate'] < 0.1:
        topology_pred = 'narrow_passage'
    elif path_metrics['edge_goal_score'] > 0.5:
        topology_pred = 'edge_goal'
    elif basic_metrics['obstacle_count'] == 0:
        topology_pred = 'open_space'
    else:
        topology_pred = 'non_blocking'
    
    # Warnings
    warnings = []
    if basic_metrics['object_goal_distance'] < 0.05:
        warnings.append('object very close to goal')
    if basic_metrics['goal_edge_distance'] < 0.02:
        warnings.append('goal very close to boundary')
    if basic_metrics['min_object_obstacle_distance'] < 0.01:
        warnings.append('object very close to obstacle')
    if contact_metrics['reachable_contact_sides'] == 0:
        warnings.append('no reachable contact sides')
    
    return {
        'difficulty_score': float(difficulty),
        'difficulty_level': difficulty_level,
        'dominant_geometric_challenge': challenges[0],
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
            invalid_reasons.append(f'obstacle {i} out of bounds: ({obs_xy[0]:.3f}, {obs_xy[1]:.3f})')
    
    # Check overlaps
    if basic_metrics['min_object_obstacle_distance'] < 0:
        invalid_reasons.append('object overlaps with obstacle')
    
    if basic_metrics['min_goal_obstacle_distance'] < 0:
        warnings.append('goal overlaps with obstacle')
    
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
    
    return {
        'geometry': {
            'object_pose': geometry['object_pose'].tolist(),
            'goal_pose': geometry['goal_pose'].tolist(),
            'ee_pose': geometry['ee_pose'].tolist(),
            'obstacle_count': geometry['obstacle_count'],
            'object_shape': geometry['object_shape'],
        },
        'basic_metrics': basic,
        'path_metrics': path,
        'contact_metrics': {k: v for k, v in contact.items() 
                           if k != 'reachable_contact_side_names'},
        'classification': classification,
        'validation': validation,
    }
