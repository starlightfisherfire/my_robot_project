#!/usr/bin/env python3
"""
Preview object shapes in 2D top-down view.

This script visualizes the geometric decomposition of all object shapes
defined in configs/object_specs.yaml using matplotlib.

Output:
    - artifacts/shape_preview.png: 2D visualization of all shapes
    - Console: detailed specifications for each shape
"""

from pathlib import Path
import yaml

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    print("matplotlib is not installed.")
    print("Install it with: pip install matplotlib")
    exit(1)


def load_object_specs(config_path: Path) -> dict:
    """Load object specifications from YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    spec_version = list(config.keys())[0]
    return config[spec_version]


def compute_T_rectangles(footprint_mm, arm_width_mm):
    """Compute rectangles for T shape (top bar + stem)."""
    W, H = footprint_mm
    w = arm_width_mm

    # Top bar: full width, arm_width height
    top_bar = {
        "center": (0, (H - w) / 2),
        "width": W,
        "height": w,
        "area": W * w,
    }

    # Stem: arm_width width, remaining height
    stem = {
        "center": (0, -(w / 2)),
        "width": w,
        "height": H - w,
        "area": w * (H - w),
    }

    return [top_bar, stem]


def compute_L_rectangles(footprint_mm, arm_width_mm):
    """Compute rectangles for L shape (horizontal bar + vertical bar)."""
    W, H = footprint_mm
    w = arm_width_mm

    # Horizontal bar: full width, arm_width height
    horiz_bar = {
        "center": (0, -(H - w) / 2),
        "width": W,
        "height": w,
        "area": W * w,
    }

    # Vertical bar: arm_width width, remaining height
    vert_bar = {
        "center": (-(W - w) / 2, w / 2),
        "width": w,
        "height": H - w,
        "area": w * (H - w),
    }

    return [horiz_bar, vert_bar]


def compute_cross_rectangles(footprint_mm, arm_width_mm):
    """
    Compute rectangles for cross shape (non-overlapping decomposition).
    - Central vertical bar: arm_width × footprint[1]
    - Left arm: arm_width × arm_width
    - Right arm: arm_width × arm_width
    """
    W, H = footprint_mm
    w = arm_width_mm

    # Central vertical bar: full height, arm_width width
    vert_bar = {
        "center": (0, 0),
        "width": w,
        "height": H,
        "area": w * H,
    }

    # Left arm: arm_width × arm_width
    left_arm = {
        "center": (-(W - w) / 2, 0),
        "width": w,
        "height": w,
        "area": w * w,
    }

    # Right arm: arm_width × arm_width
    right_arm = {
        "center": ((W - w) / 2, 0),
        "width": w,
        "height": w,
        "area": w * w,
    }

    return [vert_bar, left_arm, right_arm]


def compute_bar_rectangles(footprint_mm):
    """Compute rectangle for bar shape."""
    W, H = footprint_mm

    bar = {
        "center": (0, 0),
        "width": W,
        "height": H,
        "area": W * H,
    }

    return [bar]


def compute_square_rectangles(size_mm):
    """Compute rectangle for square pushable object shape."""
    W, H = size_mm[0], size_mm[1]

    square = {
        "center": (0, 0),
        "width": W,
        "height": H,
        "area": W * H,
    }

    return [square]


def compute_cylinder_circle(radius_mm):
    """Compute circle for cylinder pushable object shape."""
    return {
        "center": (0, 0),
        "radius": radius_mm,
        "area": 3.14159 * radius_mm ** 2,
    }


def draw_shape(ax, rectangles, shape_name, is_cylinder=False):
    """Draw a shape on the given axes."""
    if is_cylinder:
        circle = rectangles
        circ = patches.Circle(
            circle["center"],
            circle["radius"],
            linewidth=2,
            edgecolor="red",
            facecolor="lightcoral",
            alpha=0.6,
        )
        ax.add_patch(circ)

        # Draw center point
        ax.plot(0, 0, "ko", markersize=4)

        # Set equal aspect and limits
        r = circle["radius"]
        ax.set_xlim(-r * 1.5, r * 1.5)
        ax.set_ylim(-r * 1.5, r * 1.5)

    else:
        # Draw rectangles
        for i, rect in enumerate(rectangles):
            cx, cy = rect["center"]
            w, h = rect["width"], rect["height"]

            # Bottom-left corner
            x = cx - w / 2
            y = cy - h / 2

            rect_patch = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="blue",
                facecolor="lightblue",
                alpha=0.6,
            )
            ax.add_patch(rect_patch)

            # Draw center point
            ax.plot(cx, cy, "ro", markersize=4)

        # Compute bounding box
        all_x = []
        all_y = []
        for rect in rectangles:
            cx, cy = rect["center"]
            w, h = rect["width"], rect["height"]
            all_x.extend([cx - w / 2, cx + w / 2])
            all_y.extend([cy - h / 2, cy + h / 2])

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Draw bounding box
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        bbox = patches.Rectangle(
            (min_x, min_y),
            bbox_w,
            bbox_h,
            linewidth=1,
            edgecolor="gray",
            facecolor="none",
            linestyle="--",
            alpha=0.5,
        )
        ax.add_patch(bbox)

        # Set limits with margin
        margin = max(bbox_w, bbox_h) * 0.2
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_title(shape_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")


def print_shape_info(shape_name, specs, rectangles, is_cylinder=False):
    """Print detailed information about a shape."""
    print(f"\n{'=' * 70}")
    print(f"Shape: {shape_name}")
    print(f"{'=' * 70}")

    if "footprint_mm" in specs:
        print(f"Footprint: {specs['footprint_mm']} mm")
    if "size_mm" in specs:
        print(f"Size: {specs['size_mm']} mm")
    if "radius_mm" in specs:
        print(f"Radius: {specs['radius_mm']} mm")
    if "diameter_mm" in specs:
        print(f"Diameter: {specs['diameter_mm']} mm")

    thickness_mm = specs.get("thickness_mm")
    if thickness_mm is None and "height_mm" in specs:
        thickness_mm = specs["height_mm"]

    if thickness_mm:
        print(f"Thickness/Height: {thickness_mm} mm")

    if "arm_width_mm" in specs:
        print(f"Arm width: {specs['arm_width_mm']} mm")

    print(f"Nominal mass: {specs['nominal_mass_g']} g")

    if is_cylinder:
        print(f"\nCircle:")
        print(f"  Center: {rectangles['center']}")
        print(f"  Radius: {rectangles['radius']:.2f} mm")
        print(f"  Area: {rectangles['area']:.2f} mm²")
    else:
        print(f"\nRectangles ({len(rectangles)}):")
        total_area = 0
        for i, rect in enumerate(rectangles):
            print(f"  [{i}] Center: {rect['center']}, "
                  f"Width: {rect['width']:.2f} mm, "
                  f"Height: {rect['height']:.2f} mm, "
                  f"Area: {rect['area']:.2f} mm²")
            total_area += rect["area"]
        print(f"  Total area: {total_area:.2f} mm²")


def main():
    # Load config
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "configs" / "object_specs.yaml"

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    specs = load_object_specs(config_path)
    thickness_mm = specs["thickness_mm"]

    # Prepare shapes
    shapes_data = []

    # Manipulated objects
    for shape_name in ["T", "L", "cross", "bar"]:
        obj_spec = specs["manipulated_objects"][shape_name]
        obj_spec["thickness_mm"] = thickness_mm

        footprint_mm = obj_spec["footprint_mm"]
        arm_width_mm = obj_spec.get("arm_width_mm")

        if shape_name == "T":
            rectangles = compute_T_rectangles(footprint_mm, arm_width_mm)
        elif shape_name == "L":
            rectangles = compute_L_rectangles(footprint_mm, arm_width_mm)
        elif shape_name == "cross":
            rectangles = compute_cross_rectangles(footprint_mm, arm_width_mm)
        elif shape_name == "bar":
            rectangles = compute_bar_rectangles(footprint_mm)

        shapes_data.append((shape_name, obj_spec, rectangles, False))

    # Additional pushable object shapes
    square_spec = specs["obstacles"]["square"]
    square_spec["thickness_mm"] = thickness_mm
    square_rectangles = compute_square_rectangles(square_spec["size_mm"])
    shapes_data.append(("square", square_spec, square_rectangles, False))

    cylinder_spec = specs["obstacles"]["cylinder"]
    cylinder_spec["thickness_mm"] = cylinder_spec["height_mm"]
    cylinder_circle = compute_cylinder_circle(cylinder_spec["radius_mm"])
    shapes_data.append(("cylinder", cylinder_spec, cylinder_circle, True))

    # Print info
    print("=" * 70)
    print("Object Shape Preview")
    print("=" * 70)

    for shape_name, shape_spec, geom_data, is_cylinder in shapes_data:
        print_shape_info(shape_name, shape_spec, geom_data, is_cylinder)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Object Shapes - Top-Down View", fontsize=16, fontweight="bold")

    axes = axes.flatten()

    for i, (shape_name, shape_spec, geom_data, is_cylinder) in enumerate(shapes_data):
        draw_shape(axes[i], geom_data, shape_name, is_cylinder)

    plt.tight_layout()

    # Save output
    output_dir = repo_root / "artifacts"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "shape_preview.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n{'=' * 70}")
    print(f"✓ Saved visualization to: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
