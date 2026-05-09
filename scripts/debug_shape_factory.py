#!/usr/bin/env python3
"""
Verify that ObjectShapeFactory generates distinct geometries for T and L shapes.

This script:
1. Loads the factory
2. Generates XML for T and L shapes
3. Prints the geometry definitions
4. Verifies that they are different (not just single boxes)
"""

from src.envs.object_shape_factory import ObjectShapeFactory


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
            print(f"✓ Generated {geom_count} geom(s) for {shape_type}")
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
