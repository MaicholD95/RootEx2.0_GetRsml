import networkx as nx
import xml.etree.cElementTree as ET
from xml.dom import minidom
from xml.etree import ElementTree
import os.path
import matplotlib.pyplot as plt
import numpy as np
from Rsml.splines import Spline
import math
import json
import cv2 as cv
from datetime import datetime
def dict_to_rsml(metadata_info,root):
    
    # Add version element
    version = ET.SubElement(root, "version")
    version.text = str(metadata_info.get("version", ""))
    
    # Add unit element
    unit = ET.SubElement(root, "unit")
    unit.text = metadata_info.get("unit", "")
    
    # Add resolution element
    resolution = ET.SubElement(root, "resolution")
    resolution.text = str(metadata_info.get("resolution", ""))
    
    # Add last-modified element
    last_modified = ET.SubElement(root, "last-modified")
    last_modified.text = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")  # Correct format for timestamp
    
    # Add software element
    software = ET.SubElement(root, "software")
    software.text = metadata_info.get("software", "")
    
    # Add user element
    user = ET.SubElement(root, "user")
    user.text = metadata_info.get("user", "")
    
    # Add file-key element
    file_key = ET.SubElement(root, "file-key")
    file_key.text = metadata_info.get("file-key", "")
    
    # Add image element
    image = ET.SubElement(root, "image")
    
    # Add label element under image
    label = ET.SubElement(image, "label")
    label.text = metadata_info["image"].get("label", "")
       

    
    # # Convert tree to a string and return it
    # xml_string = ET.tostring(root, encoding="unicode")
    # return xml_string
    return root

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def create_rsml_from_nx(graph, metadata,final_paths,out_path):
    root = ET.Element("rsml")
    root_label = 'primary'
    seed_pos = [node for node in graph.nodes() if graph.nodes[node].get('label') == 'source']
    seeds_pts = []
    for path in seed_pos:
        seeds_pts.append(graph.nodes[path]['coord'])

    # Add metadata
    # metadata_elem = ET.SubElement(root, "metadata")
    # for key, value in metadata.items():
    #     ET.SubElement(metadata_elem, key).text = str(value)
    root = dict_to_rsml(metadata,root)

    scene_elem = ET.SubElement(root, "scene")
    #final_paths = sorted(final_paths, key=lambda x: x[0])

    for k in range(0,len(seed_pos)):
        plant_id = k
        plant_label = 'barley'
        # Add scene and plant
        plant_elem = ET.SubElement(scene_elem, "plant", ID=str(plant_id+1), label=plant_label)
        annotations = ET.SubElement(plant_elem, 'annotations')
        seed_annotation = ET.SubElement(annotations, 'annotation', name='seed position')
        ET.SubElement(seed_annotation, 'point', x=str(seeds_pts[k][1]), y=str(abs(seeds_pts[k][0]))) 


        for el in range(0,len(final_paths)):
            if final_paths[el] != []:
                first_coord = (abs(final_paths[el][0][0]),abs(final_paths[el][0][1]))

                if first_coord != [] and first_coord == tuple(seeds_pts[k]) :
                    root_id = el
                    priroot = ET.SubElement(plant_elem, 'root', ID=str(k+1)+'.'+str(el+1), label=root_label, poaccession="1")
                    geometry = ET.SubElement(priroot, 'geometry')
                    polyline = ET.SubElement(geometry, 'polyline')
                    
                    points = []
                    for i in range(0,len(final_paths[el])-1):
                        points.append((final_paths[el][i][0],final_paths[el][i][1]))
                        #points.append((graph.nodes[node]['pos'][0],-1*graph.nodes[node]['pos'][1]))
                        
                    s = Spline(points, tension = 0.5, knot_spacing = 50)
                    rootnavspline = ET.SubElement(geometry, 'rootnavspline', controlpointseparation= str(s.knot_spacing), tension=str(s.tension))
                    for c in s.knots:
                            point = ET.SubElement(rootnavspline, 'point', x=str(c[1]), y=str(c[0]))
                            
                    poly = s.polyline(sample_spacing = 1)
                    for pt in poly:
                            point = ET.SubElement(polyline, 'point', x=str(pt[1]), y=str(pt[0]))
        

    tree = ET.ElementTree(root)
    rsml_text = prettify(root)
    
    output_path = os.path.join(out_path)
    image_name = metadata['file-key']
    output_path = os.path.join(output_path, f"{image_name}.rsml")
    with open (output_path, 'w') as f:
        f.write(rsml_text)


def create_RSML(image_name,final_paths,G,RSML_output_dir):
    metadata_info = {
        "version": 1,
        "unit": "pixel",
        "resolution": 1,
        "last-modified": 1,
        "software": "RootTracer",
        "user": "pf",
        "file-key": str(image_name.replace('.jpg','')),
        "image": {
           "label": str(image_name.replace('.jpg',''))
        }
    }

    create_rsml_from_nx(G,metadata_info,final_paths,RSML_output_dir)

               
