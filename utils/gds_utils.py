import numpy as np
from sklearn.cluster import DBSCAN
import gdstk

# from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import json



def save_to_json(filename, content):
    """
    Saves the given content to a JSON file.

    Args:
        filename (str): The name (and path, if necessary) of the JSON file to save the data.
        content (dict or list): The data to be serialized and written to the JSON file.

    Behavior:
        - Opens the specified file in write mode.
        - Serializes the content into JSON format with an indentation of 4 spaces.
        - Writes the serialized JSON data to the file.

    Example:
        save_to_json('data.json', {'key': 'value'})

    Note:
        Ensure the file path exists or is writable to avoid errors.
    """
    with open(filename, 'w') as json_file:
        json.dump(content, json_file, indent=4)

def get_top_level_cells(input_file):
    """
    Reads a GDS file and retrieves the top-level cells.

    Args:
        input_file (str): The path to the GDS file to be read.

    Returns:
        dict: A dictionary where keys are the names of the top-level cells,
              and values are the corresponding cell objects.

    Behavior:
        - Reads the GDS file using gdstk.
        - Extracts the top-level cells from the GDS library.
        - Constructs a dictionary mapping each top-level cell's name to its object.

    Example:
        top_cells = get_top_level_cells("example.gds")
        # top_cells will be a dictionary with cell names as keys.

    Note:
        Ensure the input file exists and is a valid GDS file to avoid errors.
    """
    lib = gdstk.read_gds(input_file)
    top_cells = lib.top_level()
    top = {top.name: top for top in top_cells}
    return top

def flatten_selected(top_cell_dict, selected_cell_name):
    """
    Flattens the selected top-level cell.

    Args:
        top_cell_dict (dict): A dictionary of top-level cells with cell names as keys.
        selected_cell_name (str): The name of the cell to be flattened.

    Returns:
        gdstk.Cell: The flattened cell object.

    Example:
        flattened_cell = flatten_selected(top_cells, "Cell1")

    Note:
        The selected cell must exist in the dictionary; otherwise, a KeyError will occur.
    """
    selected_cell = top_cell_dict[selected_cell_name]
    return selected_cell.flatten()

def distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        point1 (array-like): Coordinates of the first point.
        point2 (array-like): Coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.

    Example:
        dist = distance([0, 0], [3, 4])  # dist = 5.0
    """
    return np.linalg.norm(point1 - point2)


def find_pads(flatten_wafer, length_standard, width_standard, strict_match=False):
    """
    Identifies rectangular or square pads on a flattened wafer based on standard dimensions.

    Args:
        flatten_wafer (gdstk.Cell): A flattened wafer cell containing polygons.
        length_standard (float): The standard length of the pads.
        width_standard (float): The standard width of the pads.
        strict_match (bool, optional): If True, ensures edges match the dimensions precisely. 
                                        Defaults to False.

    Returns:
        list: A list of bounding boxes (tuples of coordinates) for the matched pads.

    Behavior:
        - Iterates over all polygons in the flattened wafer.
        - Skips polygons that are not rectangles (not having 4 vertices).
        - Compares the polygon area to the standard pad area.
        - For strict matching:
            - Checks if the edge lengths align with the standard dimensions.
            - Ensures squares have equal lengths and widths.
            - Ensures rectangles have exactly two sides matching each dimension.
        - Adds the bounding box of matched polygons to the result list.

    Example:
        pads = find_pads(flat_wafer, 5.0, 5.0, strict_match=True)

    Notes:
        - Assumes polygons are well-defined with sequential vertices.
        - Uses `np.isclose` for floating-point comparison to handle precision issues.
    """    
    pad_coords = []
    pad_area = length_standard * width_standard
    is_square = length_standard == width_standard
    # for polygon in tqdm(flatten_wafer.polygons):
    for polygon in flatten_wafer.polygons:
        if polygon.size != 4:  # Skip polygons that are not rectangles
            continue
        if polygon.area() != pad_area:
            continue
        if not strict_match:
            pad_coords.append(polygon.bounding_box())
        else:
            vertices = polygon.points
            # calculate the length of each edge
            edge_1 = distance(vertices[0], vertices[1])
            edge_2 = distance(vertices[1], vertices[2])
            edge_3 = distance(vertices[2], vertices[3])
            edge_4 = distance(vertices[3], vertices[0])
            if is_square: # If the pad is a square
                if (np.isclose(edge_1, length_standard) and np.isclose(edge_3, length_standard) and
                    np.isclose(edge_2, width_standard) and np.isclose(edge_4, width_standard)):
                    pad_coords.append(polygon.bounding_box())
            else: # If the pad is a rectangle
                edges = [edge_1, edge_2, edge_3, edge_4]
                length_count = sum(np.isclose(edge, length_standard) for edge in edges)
                width_count = sum(np.isclose(edge, width_standard) for edge in edges)
                if length_count == 2 and width_count == 2:
                    pad_coords.append(polygon.bounding_box())

    return pad_coords

def extract_pad_centers(boxes):
    """
    Calculates the center points of bounding boxes.

    Args:
        boxes (list of tuples): A list of bounding boxes, where each box is represented
                                as a tuple of two points: ((min_x, min_y), (max_x, max_y)).

    Returns:
        np.ndarray: An array of center coordinates, where each row represents [center_x, center_y].

    Behavior:
        - Iterates over each bounding box in the input list.
        - Computes the center of the box by averaging its minimum and maximum x and y coordinates.
        - Collects the center coordinates into a list and converts it to a NumPy array.

    Example:
        boxes = [((0, 0), (2, 2)), ((4, 4), (6, 6))]
        centers = extract_pad_centers(boxes)
        # centers = np.array([[1.0, 1.0], [5.0, 5.0]])
    """
    centers = []
    for box in boxes:
        (min_x, min_y), (max_x, max_y) = box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        centers.append([center_x, center_y])
    return np.array(centers)

def cluster_pads(centers, distance_threshold):
    """
    Clusters pad center points based on their proximity using DBSCAN.

    Args:
        centers (np.ndarray): An array of center points, where each row represents [center_x, center_y].
        distance_threshold (float): The maximum distance between two points to be considered part
                                     of the same cluster.

    Returns:
        np.ndarray: An array of cluster labels, where each label corresponds to the cluster assignment
                    for the respective center point. Points with the same label belong to the same cluster.

    Behavior:
        - Uses the DBSCAN algorithm with the specified distance threshold (`eps`) to group nearby points.
        - Sets `min_samples` to 1, ensuring every point is assigned to a cluster.

    Example:
        centers = np.array([[1.0, 1.0], [1.5, 1.5], [5.0, 5.0]])
        labels = cluster_pads(centers, distance_threshold=1.0)
        # labels = np.array([0, 0, 1])
    """
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(centers)
    return clustering.labels_

def group_pads_in_sets(block_boxes, pad_set_cols=4, pad_set_rows=2):
    """
    Groups pads into sets based on grid-like arrangements.

    Args:
        block_boxes (list of tuples): A list of bounding boxes for pads, where each box is 
                                      represented as ((min_x, min_y), (max_x, max_y)).
        pad_set_cols (int, optional): Number of columns in each pad set. Defaults to 4.
        pad_set_rows (int, optional): Number of rows in each pad set. Defaults to 2.

    Returns:
        tuple:
            - list: Top-left coordinates of each pad set.
            - list: Grouped pad sets.

    Behavior:
        - Sorts bounding boxes by y-coordinate (descending) and x-coordinate (ascending).
        - Groups boxes into rows assuming a regular grid structure.
        - Ensures all rows have the same number of pads; raises an error otherwise.
        - Groups pads into sets of specified dimensions (rows x columns).
        - Extracts and returns the top-left coordinate of each set and the grouped pad sets.

    Raises:
        ArithmeticError: If not all rows have the same number of pads.

    Example:
        top_left_coords, pad_groups = group_pads_in_sets(boxes, pad_set_cols=4, pad_set_rows=2)
    """
    # Sort boxes by y coordinate (ascending) first, then by x coordinate (ascending) to form a grid structure
    sorted_boxes = sorted(block_boxes, key=lambda box: (-box[0][1], box[0][0]))

    # Group boxes by rows, assuming all boxes have the same size and are arranged in a regular grid
    grouped_pads = []
    sorted_pad_groups = []
    current_group = [sorted_boxes[0]]
    box_width = sorted_boxes[0][1][0] - sorted_boxes[0][0][0]
    box_height = sorted_boxes[0][1][1] - sorted_boxes[0][0][1]

    for i in range(1, len(sorted_boxes)):
        previous_box = current_group[-1]
        current_box = sorted_boxes[i]

        # Check if the current box is in the same row as the previous one
        if abs(current_box[0][1] - previous_box[0][1]) < box_height:
            current_group.append(current_box)
        else:
            grouped_pads.append(current_group)
            current_group = [current_box]

    # Add the last group
    if current_group:
        grouped_pads.append(current_group)

    col_num = len(current_group)
    for row in grouped_pads:
        if len(row) != col_num:
            raise ArithmeticError("Parsing error: not all rows have the same number of pads")
    
    # Group pads into sets of 8 (2x4) and extract the top-left coordinates
    # for i in range(len(grouped_pads)//2):
    #     for j in range(col_num//4):
    #         group = grouped_pads[i*2][j*4:j*4+4] + grouped_pads[i*2+1][j*4:j*4+4]
    #         sorted_pad_groups.append(group)
    for i in range(len(grouped_pads)//pad_set_rows):
        for j in range(col_num//pad_set_cols):
            group = grouped_pads[i*pad_set_rows][j*pad_set_cols:j*pad_set_cols+pad_set_cols] + grouped_pads[i*pad_set_rows+1][j*pad_set_cols:j*pad_set_cols+pad_set_cols]
            sorted_pad_groups.append(group)
    
    top_left_coords = []
    for group in sorted_pad_groups:
        top_left_box = min(group, key=lambda box: (-box[0][1], box[0][0]))
        top_left_coords.append(top_left_box[0])  # Only keep the top-left coordinate

    return top_left_coords, sorted_pad_groups

def extract_block_centers(blocks):
    """
    Calculates the center point for each block.

    Args:
        blocks (dict): A dictionary where keys are block IDs and values are lists of pad bounding boxes.

    Returns:
        dict: A dictionary with block IDs as keys and center coordinates (center_x, center_y) as values.

    Example:
        block_centers = extract_block_centers(blocks)
    """
    # Calculate the center of each block by averaging the centers of all pads in that block
    block_centers = {}
    for block_id, pads in blocks.items():
        x_coords = [((min_x + max_x) / 2) for (min_x, min_y), (max_x, max_y) in pads]
        y_coords = [((min_y + max_y) / 2) for (min_x, min_y), (max_x, max_y) in pads]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        block_centers[block_id] = (center_x, center_y)
    return block_centers

def cluster_blocks(block_centers, distance_threshold):
    """
    Clusters blocks into groups based on proximity using DBSCAN.

    Args:
        block_centers (dict): A dictionary of block centers with block IDs as keys and coordinates as values.
        distance_threshold (float): The maximum distance between two blocks to be considered in the same cluster.

    Returns:
        np.ndarray: An array of cluster labels corresponding to the block centers.

    Example:
        labels = cluster_blocks(block_centers, distance_threshold=10.0)
    """
    block_centers = np.array(list(block_centers.values()))
    # Use DBSCAN to cluster the blocks into dies
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(block_centers)
    return clustering.labels_

def sort_blocks(block_centers, primary_axis='y', secondary_axis='x', reverse_primary=True, reverse_secondary=False):
    """
    Sorts and groups blocks based on their coordinates.

    Args:
        block_centers (dict): A dictionary of block centers with block IDs as keys and coordinates as values.
        primary_axis (str, optional): The primary sorting axis ('x' or 'y'). Defaults to 'y'.
        secondary_axis (str, optional): The secondary sorting axis ('x' or 'y'). Defaults to 'x'.
        reverse_primary (bool, optional): Whether to sort the primary axis in descending order. Defaults to True.
        reverse_secondary (bool, optional): Whether to sort the secondary axis in descending order. Defaults to False.

    Returns:
        dict: A dictionary of grouped and sorted blocks by primary coordinate.

    Example:
        sorted_blocks = sort_blocks(block_centers, primary_axis='y', secondary_axis='x')
    """
    # Sort by primary axis
    sorted_by_primary = sorted(block_centers.items(), key=lambda item: item[1][1 if primary_axis == 'y' else 0], reverse=reverse_primary)
    grouped_blocks = {}
    
    # Group by primary axis
    for key, value in sorted_by_primary:
        primary_coord = value[1 if primary_axis == 'y' else 0]
        if primary_coord not in grouped_blocks:
            grouped_blocks[primary_coord] = []
        grouped_blocks[primary_coord].append((key, value))
    
    # Sort each group by secondary axis
    sorted_groups = {}
    for primary_coord, group in grouped_blocks.items():
        sorted_group = sorted(group, key=lambda item: item[1][0 if secondary_axis == 'x' else 1], reverse=reverse_secondary)
        sorted_groups[primary_coord] = sorted_group
    
    return sorted_groups

def calculate_distances(sorted_groups, axis='x'):
    """
    Calculates the distances between adjacent coordinates in sorted groups.

    Args:
        sorted_groups (dict): A dictionary of grouped items, where keys represent primary 
                              coordinates, and values are lists of items sorted by axis.
        axis (str, optional): The axis ('x' or 'y') to calculate distances along. Defaults to 'x'.

    Returns:
        dict: A dictionary where keys are the primary coordinates and values are arrays of distances 
              between consecutive items in the group.

    Example:
        distances = calculate_distances(sorted_groups, axis='x')
    """
    distances = {}
    for primary_coord, sorted_group in sorted_groups.items():
        coords = [value[0 if axis == 'x' else 1] for key, value in sorted_group]
        if len(coords) > 1:
            distances[primary_coord] = np.diff(coords)
    return distances

def calculate_general_threshold(row_distances, column_distances, z_score=0.8):
    """
    Computes a general threshold based on distances using mean and standard deviation.

    Args:
        row_distances (dict): Row-wise distances as returned by `calculate_distances`.
        column_distances (dict): Column-wise distances as returned by `calculate_distances`.
        z_score (float, optional): Multiplier for the standard deviation. Defaults to 0.8.

    Returns:
        float: The calculated general threshold.

    Example:
        threshold = calculate_general_threshold(row_distances, column_distances, z_score=1.0)
    """
    x_values = [abs(item) for sublist in list(column_distances.values()) for item in sublist]
    y_values = [abs(item) for sublist in list(row_distances.values()) for item in sublist]
    combined_values = np.concatenate([x_values, y_values])
    mean_combined = np.mean(combined_values)
    std_combined = np.std(combined_values)
    general_threshold = mean_combined + z_score * std_combined
    return general_threshold

def calculate_die_threshold(block_centers):
    """
    Calculates the distance threshold for clustering blocks into dies.

    Args:
        block_centers (dict): A dictionary of block centers with block IDs as keys and coordinates as values.

    Returns:
        float: The calculated threshold for die clustering.

    Example:
        die_threshold = calculate_die_threshold(block_centers)
    """
    sorted_rows_and_columns = sort_blocks(block_centers, primary_axis='y', secondary_axis='x', reverse_primary=True, reverse_secondary=False)
    row_distances = calculate_distances(sorted_rows_and_columns, axis='x')
    
    sorted_columns_and_rows = sort_blocks(block_centers, primary_axis='x', secondary_axis='y', reverse_primary=False, reverse_secondary=True)
    column_distances = calculate_distances(sorted_columns_and_rows, axis='y')
    
    general_threshold = calculate_general_threshold(row_distances, column_distances)
    # print("General Threshold:", general_threshold)
    return general_threshold

def group_pads_by_block(pad_coords, distance_threshold=240):
    """
    Groups pads into blocks based on clustering.

    Args:
        pad_coords (list): A list of pad bounding boxes.
        distance_threshold (float): The maximum distance between pads in the same block.

    Returns:
        tuple:
            - dict: Pads grouped into blocks.
            - dict: Structures in each block.

    Example:
        pads_in_block, structures_in_block = group_pads_by_block(pad_coords, 240)
    """
    # Extract centers of the boxes
    centers = extract_pad_centers(pad_coords)

    # Cluster pads into blocks with a specified distance threshold
    # distance_threshold = 240  # pin-to-pin distance in microns
    # distance_threshold = 480  # pin-to-pin distance in microns
    labels = cluster_pads(centers, distance_threshold)
    # Group the boxes by their cluster labels
    pads_in_block = {}
    for pad, label in zip(pad_coords, labels):
        if label not in pads_in_block:
            pads_in_block[label] = []
        pads_in_block[label].append(pad)

    # Sort blocks by descending y coordinate and ascending x coordinate
    block_items = list(pads_in_block.items())
    block_items.sort(key=lambda item: (-item[1][0][0][1], item[1][0][0][0]))

    # Renumber blocks in the sorted order
    pads_in_block = {(new_id + 1): block for new_id, (_, block) in enumerate(block_items)}
    structures_in_block = {}
    # For each block, group pads into sets of 8 and extract top-left coordinates
    for block_id, block_pads in pads_in_block.items():
        # Sort block pads by y coordinate (descending), then by x coordinate (ascending)
        block_pads = sorted(block_pads, key=lambda box: (-box[0][1], box[0][0]))
        top_left_coordinates, sorted_pad_groups = group_pads_in_sets(block_pads) # Group pads by pin sets and extract top-left coordinates
        # print(f"Block {block_id}: Top-left coordinates of groups: {top_left_coordinates}")
        # print(len(top_left_coordinates))
        structures_in_block[block_id] = top_left_coordinates
        
        """
        NOTE:   considering the amount of pads in each block, the following code is commented out,
                if the pad coordinates are needed, uncomment the code below
                IN that case the code to transform the wafer_structure to a dataframe should be updated to:
                'x': coordinate[0][0],
                'y': coordinate[0][1],
                'pads': coordinate[1] 
        """
        # coordinates_and_groups = list(zip(top_left_coordinates, sorted_pad_groups))
        # structures_in_block[block_id] = coordinates_and_groups
    return pads_in_block, structures_in_block
        


def group_blocks_by_die(blocks):
    """
    Groups blocks into dies based on proximity.

    Args:
        blocks (dict): A dictionary of blocks with IDs and their associated pads.

    Returns:
        dict: A dictionary of dies, where keys are die IDs and values are block IDs.

    Example:
        dies = group_blocks_by_die(blocks)
    """
    # Extract centers of each block
    block_centers = extract_block_centers(blocks)

    # Cluster blocks into dies with a specified distance threshold
    block_distance_threshold = calculate_die_threshold(block_centers)
    block_labels = cluster_blocks(block_centers, block_distance_threshold)

    # Group the blocks by their cluster labels to form dies
    dies = {}
    for block_id, label in zip(blocks.keys(), block_labels):
        # if label not in dies:
        #     dies[label] = []
        # dies[label].append(block_id)
        new_label = label + 1  # Shift labels to start from 1
        if new_label not in dies:
            dies[new_label] = []
        dies[new_label].append(block_id)

    blocks_in_die = {}
    # Print the grouped dies
    for die_id, die_blocks in dies.items():
        # print(f"Die {die_id}: {die_blocks}")
        blocks_in_die[die_id] = die_blocks
    return blocks_in_die



def parse_pad_coords(pad_coords, group_to_die=False, pin_spacing=240):
    """
    Parses pad coordinates to group them into blocks and optionally into dies.

    Args:
        pad_coords (list): A list of pad bounding boxes.
        group_to_die (bool, optional): Whether to group blocks into dies. Defaults to False.
        pin_spacing (float, optional): Pin spacing threshold for clustering. Defaults to 240.

    Returns:
        dict: The structured wafer layout.

    Example:
        wafer_structure = parse_pad_coords(pad_coords, group_to_die=True)
    """
    wafer_structure = {}
    pads_in_block, structures_in_block = group_pads_by_block(pad_coords, pin_spacing)

    blocks_in_die = group_blocks_by_die(pads_in_block)

    if not group_to_die:
        wafer_structure[1] = structures_in_block
    
    else:
        for die_id, die_blocks in blocks_in_die.items():
            wafer_structure[int(die_id)] = {}
            for block_id in die_blocks:
                wafer_structure[int(die_id)][int(block_id)] = structures_in_block[block_id]
    return wafer_structure




def main():
    """
    Main function to parse a GDS file and generate a wafer structure.

    Workflow:
        - Load and flatten the wafer.
        - Identify pads based on their dimensions.
        - Parse and structure the wafer layout.
        - Save the wafer structure as JSON.

    Note:
        Modify the input file and standards as necessary.
    """
    input_file = './data_structure/AgAgCl_TS_Kelvin_R2.GDS'
    length_standard = 120.0
    width_standard = 120.0
    pin_spacing = 240  # pin-to-pin distance in microns
    design_name = input_file.split('/')[-1].split('.')[0]

    top = get_top_level_cells(input_file)


    # wafer = top['wafer']
    # flatten_wafer = wafer.flatten()

    flatten_wafer = flatten_selected(top, 'wafer')

    pad_coords = find_pads(flatten_wafer, length_standard, width_standard, strict_match=False)

    np.save(f'{design_name}_pad_coords.npy', pad_coords)

    if not pad_coords:
        raise ValueError("No pads found in the GDS file")
        # pad_coords = np.load(f'{design_name}_pad_coords.npy').tolist()
    wafer_structure = parse_pad_coords(pad_coords, group_to_die=True, pin_spacing=pin_spacing)
    print(wafer_structure)


    save_to_json(f'{design_name}_wafer_structure.json', wafer_structure)

if __name__ == '__main__':
    main()

