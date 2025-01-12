import numpy as np
from sklearn.cluster import DBSCAN
import gdstk

# from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import json



def save_to_json(filename, content):
    with open(filename, 'w') as json_file:
        json.dump(content, json_file, indent=4)

def get_top_level_cells(input_file):
    lib = gdstk.read_gds(input_file)
    top_cells = lib.top_level()
    top = {top.name: top for top in top_cells}
    return top

def flatten_selected(top_cell_dict, selected_cell_name):
    selected_cell = top_cell_dict[selected_cell_name]
    return selected_cell.flatten()

def distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def find_pads(flatten_wafer, length_standard, width_standard, strict_match=False):
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
    centers = []
    for box in boxes:
        (min_x, min_y), (max_x, max_y) = box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        centers.append([center_x, center_y])
    return np.array(centers)

def cluster_pads(centers, distance_threshold):
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(centers)
    return clustering.labels_

def group_pads_in_sets(block_boxes, pad_set_cols=4, pad_set_rows=2):
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
    block_centers = np.array(list(block_centers.values()))
    # Use DBSCAN to cluster the blocks into dies
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(block_centers)
    return clustering.labels_

def sort_blocks(block_centers, primary_axis='y', secondary_axis='x', reverse_primary=True, reverse_secondary=False):
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
    distances = {}
    for primary_coord, sorted_group in sorted_groups.items():
        coords = [value[0 if axis == 'x' else 1] for key, value in sorted_group]
        if len(coords) > 1:
            distances[primary_coord] = np.diff(coords)
    return distances

def calculate_general_threshold(row_distances, column_distances, z_score=0.8):
    x_values = [abs(item) for sublist in list(column_distances.values()) for item in sublist]
    y_values = [abs(item) for sublist in list(row_distances.values()) for item in sublist]
    combined_values = np.concatenate([x_values, y_values])
    mean_combined = np.mean(combined_values)
    std_combined = np.std(combined_values)
    general_threshold = mean_combined + z_score * std_combined
    return general_threshold

def calculate_die_threshold(block_centers):
    sorted_rows_and_columns = sort_blocks(block_centers, primary_axis='y', secondary_axis='x', reverse_primary=True, reverse_secondary=False)
    row_distances = calculate_distances(sorted_rows_and_columns, axis='x')
    
    sorted_columns_and_rows = sort_blocks(block_centers, primary_axis='x', secondary_axis='y', reverse_primary=False, reverse_secondary=True)
    column_distances = calculate_distances(sorted_columns_and_rows, axis='y')
    
    general_threshold = calculate_general_threshold(row_distances, column_distances)
    # print("General Threshold:", general_threshold)
    return general_threshold

def group_pads_by_block(pad_coords, distance_threshold=240):
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

