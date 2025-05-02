import os
import sys

# Minimal test for Lanelet2 map loading

# Known problematic values from recording 1 (location 4)
MAP_FILE_PATH = "/home/godwin/Downloads/sensor_fusion_dst_package_v1.1/sensor_fusion_dst/inD-dataset-v1.1/maps/lanelets/04_aseag/location4.osm"
UTM_ORIGIN_X = 297631.3187
UTM_ORIGIN_Y = 5629917.34465

print(f"Attempting to import Lanelet2...")
try:
    import lanelet2
    from lanelet2.projection import UtmProjector
    from lanelet2.io import Origin, load
    print("Lanelet2 imported successfully.")
    LANELET2_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Failed to import Lanelet2: {e}")
    LANELET2_AVAILABLE = False
    sys.exit(1)

if LANELET2_AVAILABLE:
    print(f"\nChecking map file existence: {MAP_FILE_PATH}")
    if not os.path.exists(MAP_FILE_PATH):
        print("ERROR: Map file does not exist.")
        sys.exit(1)
    else:
        print("Map file found.")

    print("\nAttempting to create UtmProjector and load map...")
    projector = None
    lanelet_map = None
    try:
        print(f"Creating Origin with x={UTM_ORIGIN_X}, y={UTM_ORIGIN_Y}")
        # This is the suspected problematic line
        origin = Origin(UTM_ORIGIN_X, UTM_ORIGIN_Y)
        print("Origin created.")
        
        print("Creating UtmProjector...")
        projector = UtmProjector(origin)
        print("UtmProjector created.")
        
        print(f"Loading map: {MAP_FILE_PATH}")
        lanelet_map = load(MAP_FILE_PATH, projector)
        print("Map loaded successfully!")
        
    except Exception as e:
        print(f"\n--- EXCEPTION CAUGHT --- ")
        print(f"ERROR: {e}")
        print("------------------------")
        
    print("\nFinal check:")
    print(f"Projector is None: {projector is None}")
    print(f"Lanelet map is None: {lanelet_map is None}")

