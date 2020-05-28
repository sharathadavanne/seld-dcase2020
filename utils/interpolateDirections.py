import math
import numpy as np

def interpolateDirections(dir1_xyz, dir2_xyz, N):
#
# interpolate two direction vectors by performing spherical (rotational) 
# linear interpolation, generating direction vectors in between the two,
# across the great circle passing from them. The angle between the two is
# divided in N intervals, hence N+1 vectors are returned, including the two
# original ones.
#
# dir1_xyz: 	numpy array, star vector
# dir2_xyz: 	numpy array, end vector
# N: 		integer, number of intervals to interpolate

	angle12 = math.acos(dir1_xyz.dot(dir2_xyz))
	cross12 = np.cross(dir1_xyz, dir2_xyz)
	purpvec12 = cross12/math.sqrt(cross12.dot(cross12))

	dtheta = angle12/N
	psi = np.linspace(0,angle12,N+1)
	interp_dirs_xyz = []
	for x in psi:
		dirn_xyz = dir1_xyz * math.cos(x) + np.cross(purpvec12,dir1_xyz)*math.sin(x) + purpvec12.dot(np.cross(purpvec12,dir1_xyz))*(1-math.cos(x))
		interp_dirs_xyz.append(dirn_xyz)
	
        return np.array(interp_dirs_xyz)
