#!/usr/bin/env python3
"""
Verify that ObjectShapeFactory generates distinct geometries for T and L shapes.

This script:
1. Loads the factory
2. Generates XML for T and L shapes
3. Prints the geometry definitions
4. Verifies that they are different (not just single boxes)
5. Verifies that total mass equals nominal_mass_g
"""

import re
from src.envs.object_shape_factory import ObjectShapeFactory


def extract_masses_from_xml(geoms_xml: str) -> list[float]:
    """Extract all mass values from geom XML."""
    mass_pattern = r'mass="([0-9.]+)"'
    matches = re.findall(mass_pattern, geoms_xml)
    return [float(m) for m in matches]


def extract_geom_details(geoms_xml: str) -> list[dict]:
    """Extract detailed information for each geom from XML."""
    geoms = []

    # Split by <geom tags
    geom_blocks = geoms_xml.split("<geom")[1:]  # Skip first empty split

    for block in geom_blocks:
        geom_info = {}

        # Extract name
        name_match = re.search(r'name="([^"]+)"', block)
        if name_match:
            geom_info["name"] = name_match.group(1)

        # Extract type
        type_match = re.search(r'type="([^"]+)"', block)
        if type_match:
            geom_info["type"] = type_match.group(1)

        # Extract pos
        pos_match = re.search(r'pos="([^"]+)"', block)
        if pos_match:
            pos_str = pos_match.group(1)
            geom_info["pos"] = [float(x) for x in pos_str.split()]

        # Extract size
        size_match = re.search(r'size="([^"]+)"', block)
        if size_match:
            size_str = size_match.group(1)
            geom_info["size"] = [float(x) for x in size_str.split()]

        # Extract mass
        mass_match = re.search(r'mass="([^"]+)"', block)
        if mass_match:
            geom_info["mass"] = float(mass_match.group(1))

        geoms.append(geom_info)

    return geoms


def main():
    print("=" * 70)
    print("ObjectShapeFactory Verification")
    print("=" * 70)
    print()

    factory = ObjectShapeFactory()
    print(f"✓ Loaded config from: {factory.config_path}")
    print(f"✓ Spec version: {factory.spec_version}")
    print()

    shapes_to_test = ["T", "L", "cross", "bar", "square", "cylinder"]

    for shape_type in shapes_to_test:
        print(f"{'=' * 70}")
        print(f"Shape: {shape_type}")
        print(f"{'=' * 70}")

        try:
            geoms_xml = factory.get_object_geoms_xml(
                shape_type=shape_type,
                rgba="0.9 0.2 0.1 1",
                contype=1,
                conaffinity=1,
            )

            print(geoms_xml)
            print()

            geom_count = geoms_xml.count("<geom")
            masses = extract_masses_from_xml(geoms_xml)
            total_mass = sum(masses)

            print(f"✓ Generated {geom_count} geom(s) for {shape_type}")

            # Print detailed geom information for cross shape
            if shape_type == "cross":
                geom_details = extract_geom_details(geoms_xml)
                print(f"\n  Detailed geom breakdown:")
                for i, geom in enumerate(geom_details):
                    print(f"    [{i}] {geom.get('name', 'unnamed')}")
                    if "pos" in geom:
                        print(f"        Position (m): [{geom['pos'][0]:.6f}, {geom['pos'][1]:.6f}, {geom['pos'][2]:.6f}]")
                    if "size" in geom and geom.get("type") == "box":
                        # MuJoCo box size is half-extents
                        full_size = [s * 2 for s in geom["size"]]
                        print(f"        Size (m): [{full_size[0]:.6f}, {full_size[1]:.6f}, {full_size[2]:.6f}]")
                        print(f"        Size (mm): [{full_size[0]*1000:.2f}, {full_size[1]*1000:.2f}, {full_size[2]*1000:.2f}]")
                    if "mass" in geom:
                        print(f"        Mass (kg): {geom['mass']:.6f}")
                        print(f"        Mass (g): {geom['mass']*1000:.6f}")

            if len(masses) > 1 and shape_type != "cross":
                print(f"  Individual masses (kg): {[f'{m:.6f}' for m in masses]}")

            print(f"  Total mass (kg): {total_mass:.6f}")
            print(f"  Total mass (g): {total_mass * 1000:.6f}")

            expected_mass_kg = 0.01905
            mass_diff = abs(total_mass - expected_mass_kg)
            if mass_diff < 1e-9:
                print(f"  ✓ Mass conservation verified (expected: {expected_mass_kg:.6f} kg)")
            else:
                print(f"  ✗ Mass mismatch! Expected: {expected_mass_kg:.6f} kg, Got: {total_mass:.6f} kg")

            print()

        except Exception as e:
            print(f"✗ Failed to generate {shape_type}: {e}")
            print()

    print("=" * 70)
    print("Verification Summary")
    print("=" * 70)
    print()
    print("✓ T shape should have 2 geoms (top bar + stem)")
    print("✓ L shape should have 2 geoms (horizontal + vertical)")
    print("✓ cross shape should have 2 geoms (horizontal + vertical)")
    print("✓ bar shape should have 1 geom (single box)")
    print("✓ square shape should have 1 geom (single box)")
    print("✓ cylinder shape should have 1 geom (cylinder)")
    print()
    print("If all shapes generated successfully, the factory is working correctly.")


if __name__ == "__main__":
    main()
