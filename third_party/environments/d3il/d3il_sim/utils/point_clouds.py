import numpy as np
import open3d


class PcVisualizer:
    def __init__(self):
        self.point_cloud_visualizer = None

    def visualize_point_clouds(self, points, colors):
        colors = colors[:, :3]
        if self.point_cloud_visualizer is None:
            self.point_cloud_visualizer = open3d.visualization.Visualizer()
            self.point_cloud_visualizer.create_window()
            self.point_cloud_visualizer_helper = open3d.geometry.PointCloud()
            self.point_cloud_visualizer_helper.points = open3d.utility.Vector3dVector(
                points
            )
            self.point_cloud_visualizer_helper.colors = open3d.utility.Vector3dVector(
                colors
            )
            self.point_cloud_visualizer.add_geometry(self.point_cloud_visualizer_helper)
        # open3d.visualization.draw_geometries([self.visualizer])
        else:
            # self.test.remove_geometry(self.visualizer)
            self.point_cloud_visualizer_helper.points = open3d.utility.Vector3dVector(
                points
            )
            self.point_cloud_visualizer_helper.colors = open3d.utility.Vector3dVector(
                colors
            )
            # open3d.visualization.draw_geometries([self.visualizer])
            self.point_cloud_visualizer.add_geometry(self.point_cloud_visualizer_helper)

        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        self.point_cloud_visualizer.add_geometry(mesh_frame)
        self.point_cloud_visualizer.run()
        self.point_cloud_visualizer.destroy_window()
        self.point_cloud_visualizer = None
        self.point_cloud_visualizer_helper = None


def rgb_float_to_int(rgb_float):
    """
    Covert rgb value from [0, 1] to [0, 255]
    Args:
        rgb_float: rgb array

    Returns:
        int rgb values
    """
    return (rgb_float * 255).astype(dtype=np.uint32)


def rgb_array_to_uint32(rgb_array):
    """
    Pack 3 rgb values into 1 uint32 integer value using binary operations
    Args:
        rgb_array: array [num_samples, num_data=3]

    Returns:
        packed rgb integer value: [num_samples], dtype=np.uint32
        From left to right:
            0-8 bits: place holders (can be extent to additional channel)
            9-16 bits: red
            17-24 bits: green
            25-32 bits: blue
    """
    rgb32 = np.zeros(rgb_array.shape[0], dtype=np.uint32)
    rgb_array = rgb_array.astype(dtype=np.uint32)
    rgb32[:] = (
        np.left_shift(rgb_array[:, 0], 16)
        + np.left_shift(rgb_array[:, 1], 8)
        + rgb_array[:, 2]
    )
    return rgb32
