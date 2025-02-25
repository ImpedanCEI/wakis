# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import numpy as np
import re

def extract_colors_from_stp(stp_file):
    lines = np.genfromtxt(stp_file, dtype=str, delimiter="\n", encoding="utf-8", comments=None)

    solids = []
    colors = []
    color_dict = {}

    # Regex patterns
    solid_pattern = re.compile(r"#(\d+)=MANIFOLD_SOLID_BREP\('([^']+)'.*;")
    color_pattern = re.compile(r"#(\d+)=COLOUR_RGB\('?',([\d.]+),([\d.]+),([\d.]+)\);")

    # Extract colors
    for line in lines:
        color_match = color_pattern.search(line)
        if color_match:
            r, g, b = float(color_match.group(2)), float(color_match.group(3)), float(color_match.group(4))
            colors.append([r, g, b])

    # Extract solids
    for line in lines:
        solid_match = solid_pattern.search(line)
        if solid_match:
            solid_name = solid_match.group(2)
            solids.append(solid_name)

    for i in range(len(solids)):
        color_dict[solids[i]] = colors[i]

    return color_dict

def extract_materials_from_stp(stp_file):
    lines = np.genfromtxt(stp_file, dtype=str, delimiter="\n", encoding="utf-8", comments=None)

    solid_dict = {}
    material_dict = {}

    # Regex patterns
    solid_pattern = re.compile(r"#(\d+)=MANIFOLD_SOLID_BREP\('([^']+)'.*;")
    material_pattern = re.compile(r"#(\d+)=PRESENTATION_LAYER_ASSIGNMENT\('([^']+)','[^']+',\(#([\d#,]+)\)\);")

    # Extract solids
    for line in lines:
        solid_match = solid_pattern.search(line)
        if solid_match:
            solid_number = int(solid_match.group(1))
            solid_name = solid_match.group(2)
            solid_dict[solid_number] = solid_name

    # Extract materials
    for line in lines:
        material_match = material_pattern.search(line)
        if material_match:
            material_name = material_match.group(2)
            solid_numbers = [int(num.strip("#")) for num in material_match.group(3).split(',')] # Extract numbers as a list
            
            for solid_number in solid_numbers:
                if solid_number in solid_dict:  # Only assign if it's a known solid
                    material_dict[solid_dict[solid_number]] = material_name

    return material_dict

def extract_solids_from_stp(stp_file):
    solids, materials = extract_names_from_stp(stp_file)
    stl_solids = {}
    for i in range(len(list(solids.keys()))):
        solid = solids[list(solids.keys())[i]]
        mat = materials[list(solids.keys())[i]]

        # Remove problematic characters
        solid_re = re.sub(r'[^a-zA-Z0-9_-]', '-', solid)
        mat_re = re.sub(r'[^a-zA-Z0-9_-]', '-', mat)
        name = f'{str(i).zfill(3)}_{solid_re}_{mat_re}'
        stl_solids[solid] = name + '.stl'

    return stl_solids

def extract_names_from_stp(stp_file):
    """
    Extracts solid names and their corresponding materials from a STEP (.stp) file.

    This function parses a given STEP file to identify solid objects and their assigned materials.
    The solid names are extracted from `MANIFOLD_SOLID_BREP` statements, while the materials are 
    linked via `PRESENTATION_LAYER_ASSIGNMENT` statements.

    Args:
        stp_file (str): Path to the STEP (.stp) file.

    Returns:
        tuple[dict[int, str], dict[int, str]]: 
            - A dictionary mapping solid IDs to their names.
            - A dictionary mapping solid IDs to their corresponding material names.

    Example:
        >>> solids, materials = extract_names_from_stp("example.stp")
        >>> print(solids)
        {37: "Vacuum|Half_cell_dx", 39: "Be window left"}
        >>> print(materials)
        {37: "Vacuum", 39: "Berillium"}
    """
    lines = np.genfromtxt(stp_file, dtype=str, delimiter="\n", encoding="utf-8", comments=None)

    solid_dict = {}
    material_dict = {}

    # Regex patterns
    solid_pattern = re.compile(r"#(\d+)=MANIFOLD_SOLID_BREP\('([^']+)'.*;")
    material_pattern = re.compile(r"#(\d+)=PRESENTATION_LAYER_ASSIGNMENT\('([^']+)','[^']+',\(#([\d#,]+)\)\);")

    # Extract solids
    for line in lines:
        solid_match = solid_pattern.search(line)
        if solid_match:
            solid_number = int(solid_match.group(1))
            solid_name = solid_match.group(2)
            solid_dict[solid_number] = solid_name

    # Extract materials
    for line in lines:
        material_match = material_pattern.search(line)
        if material_match:
            material_name = material_match.group(2)
            solid_numbers = [int(num.strip("#")) for num in material_match.group(3).split(',')] # Extract numbers as a list
            
            for solid_number in solid_numbers:
                if solid_number in solid_dict:  # Only assign if it's a known solid
                    material_dict[solid_number] = material_name

    return solid_dict, material_dict

def generate_stl_solids_from_stp(stp_file):
    """
    Extracts solid objects from a STEP (.stp) file and exports them as STL files.

    This function:
    - Imports the STEP file using `cadquery`.
    - Extracts solid names and their materials using `extract_names_from_stp()`.
    - Sanitizes solid and material names by replacing problematic characters.
    - Saves each solid as an STL file in the current directory.

    Args:
        stp_file (str): Path to the STEP (.stp) file.

    Raises:
        Exception: If `cadquery` is not installed, it prompts the user to install it.

    Example:
        >>> extract_stl_solids_from_stp("example.stp")
        000_Vacuum-Half_cell_dx_Vacuum.stl
        001_Be_window_left_Berillium.stl
    """
        
    try:
        import cadquery as cq
    except:
        raise Exception('''This function needs the open-source package `cadquery`
                        To install it in a conda environment do: 
                        
                        `pip install cadquery`

                        [!] We recommend having a dedicated conda environment to avoid version issues
                        ''')
    
    stp = cq.importers.importStep(stp_file)
    solid_dict = extract_solids_from_stp(stp_file)

    print(f"Generating stl from file: {stp_file}... ")
    for i, obj in enumerate(stp.objects[0]):
        name = solid_dict[list(solid_dict.keys())[i]]
        print(name+".stl")
        obj.exportStl(name+".stl") 

    return solid_dict