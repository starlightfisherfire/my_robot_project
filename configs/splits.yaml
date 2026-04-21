version: "0.3"

description: >
  Split protocol for Paper 1: causality-aware object-level high-level
  representations under fixed CEM-MPC for structural OOD generalization.
  Simulation is used for mechanism identification and ablation.
  Real robot experiments are used for zero-shot sim-to-real transfer,
  real-ID adaptation, and real-world OOD validation.

protocols:
  sim_mechanism:
    description: >
      Main controlled simulation protocol. Models are trained on simulation ID
      layouts and evaluated on held-out simulation layout and shape OOD splits.

  real_zero_shot_transfer:
    description: >
      Models trained only on simulation ID data are evaluated directly on real
      robot ID and OOD splits. This evaluates both sim-to-real transfer and
      structural OOD under domain shift.

  real_id_adapted_transfer:
    description: >
      Models are first trained on simulation ID data and then adapted using a
      small real robot ID adaptation set. The adaptation set only contains
      T-shaped objects and ID layouts. Real OOD splits are strictly held out.

ood_axes:
  layout_ood:
    id_families:
      - "open_space"
      - "mild_offset"
    ood_families:
      - "blocking"
      - "narrow_passage"
      - "edge_goal"

  shape_ood:
    train_shapes:
      - "T"
    ood_shapes:
      - "L"

splits:
  train_sim_id:
    role: "train"
    source: "sim"
    description: "Simulation ID training split."
    object_shapes:
      - "T"
    layout_families:
      - "open_space"
      - "mild_offset"
    allowed_for:
      - "sim_mechanism"
      - "real_zero_shot_transfer"
      - "real_id_adapted_transfer"

  val_sim_id:
    role: "validation"
    source: "sim"
    description: "Simulation ID validation split for model selection and early stopping."
    object_shapes:
      - "T"
    layout_families:
      - "open_space"
      - "mild_offset"
    allowed_for:
      - "sim_mechanism"
      - "real_zero_shot_transfer"
      - "real_id_adapted_transfer"

  test_sim_id:
    role: "test"
    source: "sim"
    description: "Simulation ID test split."
    object_shapes:
      - "T"
    layout_families:
      - "open_space"
      - "mild_offset"
    allowed_for:
      - "sim_mechanism"

  test_sim_layout_ood_blocking:
    role: "test"
    source: "sim"
    description: "Simulation layout OOD: obstacle blocks the direct path."
    object_shapes:
      - "T"
    layout_families:
      - "blocking"
    held_out_factor: "layout"
    allowed_for:
      - "sim_mechanism"

  test_sim_layout_ood_narrow_passage:
    role: "test"
    source: "sim"
    description: "Simulation layout OOD: narrow passage."
    object_shapes:
      - "T"
    layout_families:
      - "narrow_passage"
    held_out_factor: "layout"
    allowed_for:
      - "sim_mechanism"

  test_sim_layout_ood_edge_goal:
    role: "test"
    source: "sim"
    description: "Simulation layout OOD: goal near workspace boundary."
    object_shapes:
      - "T"
    layout_families:
      - "edge_goal"
    held_out_factor: "layout"
    allowed_for:
      - "sim_mechanism"

  test_sim_shape_ood_L:
    role: "test"
    source: "sim"
    description: "Simulation shape OOD: train on T-shape, test on L-shape."
    object_shapes:
      - "L"
    layout_families:
      - "open_space"
      - "mild_offset"
      - "blocking"
    held_out_factor: "shape"
    allowed_for:
      - "sim_mechanism"

  calibrate_real_id:
    role: "calibration"
    source: "real"
    description: >
      Real robot calibration split for camera calibration, coordinate alignment,
      workspace boundary checking, and basic system sanity tests. This split
      should not be used for final OOD evaluation.
    object_shapes:
      - "T"
    layout_families:
      - "open_space"
    allowed_for:
      - "real_zero_shot_transfer"
      - "real_id_adapted_transfer"

  adapt_real_id:
    role: "adaptation"
    source: "real"
    description: >
      Small real robot ID adaptation split. It may be used for sim-to-real
      adaptation, normalization fitting, light model fine-tuning, dynamics-head
      adaptation, or real residual calibration. It contains only T-shaped objects
      and ID layouts, so it does not leak layout OOD or shape OOD.
    object_shapes:
      - "T"
    layout_families:
      - "open_space"
      - "mild_offset"
    allowed_for:
      - "real_id_adapted_transfer"

  val_real_id:
    role: "validation"
    source: "real"
    description: >
      Real robot ID validation split. Used to check whether the adapted system
      works on real ID layouts. Not used for OOD test reporting.
    object_shapes:
      - "T"
    layout_families:
      - "open_space"
      - "mild_offset"
    allowed_for:
      - "real_id_adapted_transfer"

  test_real_id:
    role: "test"
    source: "real"
    description: >
      Real robot ID test split. In the zero-shot protocol, this measures basic
      sim-to-real transfer. In the adapted protocol, this measures real ID
      performance after real-ID adaptation.
    object_shapes:
      - "T"
    layout_families:
      - "open_space"
      - "mild_offset"
    allowed_for:
      - "real_zero_shot_transfer"
      - "real_id_adapted_transfer"

  test_real_layout_ood_blocking:
    role: "test"
    source: "real"
    description: >
      Real robot layout OOD: obstacle blocks the direct path. In the zero-shot
      protocol, this evaluates sim-to-real plus structural OOD. In the adapted
      protocol, this evaluates real-ID-adapted OOD generalization.
    object_shapes:
      - "T"
    layout_families:
      - "blocking"
    held_out_factor: "layout"
    allowed_for:
      - "real_zero_shot_transfer"
      - "real_id_adapted_transfer"

  test_real_layout_ood_narrow_passage:
    role: "test"
    source: "real"
    description: >
      Real robot layout OOD: narrow passage. In the zero-shot protocol, this
      evaluates sim-to-real plus structural OOD. In the adapted protocol, this
      evaluates real-ID-adapted OOD generalization.
    object_shapes:
      - "T"
    layout_families:
      - "narrow_passage"
    held_out_factor: "layout"
    allowed_for:
      - "real_zero_shot_transfer"
      - "real_id_adapted_transfer"

  test_real_layout_ood_edge_goal:
    role: "test"
    source: "real"
    description: >
      Real robot layout OOD: goal near workspace boundary. In the zero-shot
      protocol, this evaluates sim-to-real plus structural OOD. In the adapted
      protocol, this evaluates real-ID-adapted OOD generalization.
    object_shapes:
      - "T"
    layout_families:
      - "edge_goal"
    held_out_factor: "layout"
    allowed_for:
      - "real_zero_shot_transfer"
      - "real_id_adapted_transfer"

  test_real_shape_ood_L:
    role: "test"
    source: "real"
    description: >
      Real robot shape OOD: train and adapt on T-shape, test on L-shape.
    object_shapes:
      - "L"
    layout_families:
      - "open_space"
      - "mild_offset"
      - "blocking"
    held_out_factor: "shape"
    allowed_for:
      - "real_zero_shot_transfer"
      - "real_id_adapted_transfer"

leakage_rules:
  - "No layout OOD family may appear in train_sim_id, adapt_real_id, val_real_id, or test_real_id."
  - "No L-shaped object may appear in train_sim_id, adapt_real_id, val_real_id, or test_real_id."
  - "Real OOD test splits must not be used for model selection, hyperparameter tuning, normalization fitting, early stopping, or adaptation."
  - "All compared methods must use the same train_sim_id data budget."
  - "All real-ID-adapted methods must use the same adapt_real_id data budget."
  - "If real adaptation is used, results must be reported as real-ID-adapted OOD generalization, not pure zero-shot sim-to-real transfer."
  - "Zero-shot real OOD results must be interpreted as a compound evaluation of sim-to-real transfer plus structural OOD."

reporting:
  main_sim_results:
    protocol: "sim_mechanism"
    train_split: "train_sim_id"
    validation_split: "val_sim_id"
    id_split: "test_sim_id"
    ood_splits:
      - "test_sim_layout_ood_blocking"
      - "test_sim_layout_ood_narrow_passage"
      - "test_sim_layout_ood_edge_goal"
      - "test_sim_shape_ood_L"

  real_zero_shot_results:
    protocol: "real_zero_shot_transfer"
    train_split: "train_sim_id"
    adaptation_split: null
    id_split: "test_real_id"
    ood_splits:
      - "test_real_layout_ood_blocking"
      - "test_real_layout_ood_narrow_passage"
      - "test_real_layout_ood_edge_goal"
      - "test_real_shape_ood_L"

  real_id_adapted_results:
    protocol: "real_id_adapted_transfer"
    train_split: "train_sim_id"
    adaptation_split: "adapt_real_id"
    validation_split: "val_real_id"
    id_split: "test_real_id"
    ood_splits:
      - "test_real_layout_ood_blocking"
      - "test_real_layout_ood_narrow_passage"
      - "test_real_layout_ood_edge_goal"
      - "test_real_shape_ood_L"