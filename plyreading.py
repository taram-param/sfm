import open3d as o3d
import numpy as np

plydata = o3d.io.read_point_cloud('results/fountain-P11/point-clouds/cloud_11_view.ply')
print(plydata)
print(np.asarray(plydata.points))
o3d.visualization.draw_geometries([plydata])



