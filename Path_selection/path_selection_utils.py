
import cv2
def print_single_path_on_image(image,path,graph,name,color):
    """
    Print a single path on an image.

    Args:
        image (np.array): The image on which to print the path.
        path (list): The list of nodes in the path.

    Returns:
        np.array: The image with the path printed on it.
    """
    # Create a copy of the image to draw on
    image_with_path = image.copy()
    # Get the coordinates of the nodes in the path
    path_coords = [graph.nodes[node]['coord'] for node in path]
    # Draw the path on the image
    for i in range(len(path_coords) - 1):
        cv2.line(
            image_with_path,
            (int(path_coords[i][1]), int(path_coords[i][0])),
            (int(path_coords[i + 1][1]), int(path_coords[i + 1][0])),
            color,
            3
        )
    cv2.imwrite(f'{name}.png', image_with_path)
    return image_with_path