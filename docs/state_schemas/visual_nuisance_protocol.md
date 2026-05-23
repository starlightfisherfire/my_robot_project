# visual_nuisance_protocol.md

## Purpose
Define visual nuisance variables and protocol for invariance testing.

## Nuisance Variables

| Variable | Type | Range | Randomization |
|----------|------|-------|---------------|
| object_color_rgb | categorical | 8 colors | uniform |
| obstacle_color_rgb | categorical | 8 colors | uniform |
| table_texture_id | categorical | 4 textures | uniform |
| background_texture_id | categorical | 4 textures | uniform |
| light_position | continuous | [0.5, 1.5]m radius | uniform |
| light_direction | continuous | [30°, 60°] elevation | uniform |
| ambient_light | continuous | [0.1, 0.5] | uniform |
| diffuse_light | continuous | [0.5, 1.0] | uniform |
| specular_light | continuous | [0.0, 0.3] | uniform |
| shadow_strength | continuous | [0.0, 1.0] | uniform |
| camera_fovy | continuous | [45°, 60°] | uniform |
| render_noise_seed | integer | [0, 2^16] | uniform |

## Protocol
1. Domain randomization: During data collection, randomize nuisance per episode
2. Invariance test: Train on randomized, test on held-out nuisance values
3. z_nuisance diagnostic: Use causality_aware encoder's z_nuisance slot to diagnose

## Usage
- ALLOWED: Domain randomization, invariance test, z_nuisance diagnostic
- FORBIDDEN: As main dynamics state input for the primary claim
