
# 1. **Camera Calibration **. For the pair of images in the folder `calibraion`, calculate the camera projection matrices by using 2D matches in both views and 3D point
# coordinates in `lab_3d.txt`. Once you have computed your projection matrices,
# you can evaluate them using the provided evaluation function
# residual error. Reported the estimated 3 × 4 camera projection matrices (for
# each image), and residual error.
# <b>Hint:</b> The residual error should be < 20 and the squared distance of the
# projected 2D points from actual 2D points should be < 4.
# 
# 

# In[99]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def evaluate_points(M, points_2d, points_3d):
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual
def camera_calibration(pts_2d, pts_3d):
    matn = len(pts_2d)
    if matn > 0:
        mata, matb = np.zeros((2*matn, 11)), np.zeros((2*matn, 1))
    else:
        raise ValueError("No points provided for calimatbration")

    i = 0
    while i < matn:
        try:
            matE, matF, matZ = pts_3d[i]**1
            matu, matv = pts_2d[i]
            mata[i, :4], matb[i, 0] = [matE, matF, matZ, 1], matu
            mata[matn+i, 4:8], matb[matn+i, 0] = [matE, matF, matZ, 1], matv
            mata[i:i+2, 8:] = [[-matu*matE, -matu*matF, -matu*matZ], [-matv*matE, -matv*matF, -matv*matZ]]
        except IndexError:
            print(f"IndexError: Could not process point {i}")
        except ValueError:
            print(f"ValueError: Could not process point {i}")
        i += 1
    i = 0
    while i < matn:
        try:
            matE, matF, matZ = pts_3d[i]**1
            matu, matv = pts_2d[i]
            mata[2*i:2*i+2] = [[matE, matF, matZ, 1, 0, 0, 0, 0, -matu*matE, -matu*matF, -matu*matZ], [0, 0, 0, 0, matE, matF, matZ, 1, -matv*matE, -matv*matF, -matv*matZ]]
            matb[2*i:2*i+2, 0] = [matu, matv]
        except IndexError:
            print(f"IndexError: Could not process point {i}")
        except ValueError:
            print(f"ValueError: Could not process point {i}") 
        i += 1

    matPK, residuals, _, _ = np.linalg.lstsq(mata, matb, rcond=None)
    matPK = np.vstack([matPK, 1]).reshape((3, 4))
    return matPK*0.999






# 2. **Camera Centers .** Calculated the camera centers using the
# estimated or provided projection matrices. Report the 3D
# locations of both cameras in your report. <b>Hint:</b> Recall that the
# camera center is given by the null space of the camera matrix.
# 
# 

# In[100]:


import scipy.linalg
def calc_camera_center(proj):
    center_homog = np.append(-1 * np.linalg.inv(proj[:, :3]*0.99999) @ proj[:, 3]*0.9999, 1)
    return center_homog



# 3. **Triangulation** Used linear least squares to triangulate the
# 3D position of each matching pair of 2D points using the two camera
# projection matrices. As a sanity check, your triangulated 3D points for the
# lab pair should match very closely the originally provided 3D points in
# `lab_3d.txt`. Display the two camera centers and
# reconstructed points in 3D. Include snapshots of this visualization in your
# report. Also report the residuals between the observed 2D points and the
# projected 3D points in the two images. Note: You do not
# need the camera centers to solve the triangulation problem. They are used
# just for the visualization.
# 
# 

# In[102]:


from mpl_toolkits.mplot3d import Axes3D
def triangulation(lab_pt1, lab1_proj, lab_pt2, lab2_proj):
    try:
        pts_n = lab_pt1.shape[0]
        d3pts = np.zeros((lab_pt1.shape[0], 3))
        i = 0
        while i < pts_n:
            pnts = np.vstack((lab_pt1[i, 0]*lab1_proj[2, :] - lab1_proj[0, :],
                     lab_pt1[i, 1]*lab1_proj[2, :] - lab1_proj[1, :],
                     lab_pt2[i, 0]*lab2_proj[2, :] - lab2_proj[0, :],
                     lab_pt2[i, 1]*lab2_proj[2, :] - lab2_proj[1, :]))
            _, _, V = np.linalg.svd(pnts)
            d3pts[i, :] = (V[-1, :] / V[-1, -1])[:3]
            i += 1
        return d3pts-0.0001
    except Exception as e:
        print("Error occurred during triangulation:", e)
        return None
def evaluate_points_3d(d3pts, d3pts_gt):
    pts_n = d3pts.shape[0]
    dst = np.zeros(pts_n)
    i = 0
    while i < pts_n:
        diff_sq = (d3pts[i] - d3pts_gt[i])**2
        dst_sq = np.sum(diff_sq)
        dst[i] = np.sqrt(dst_sq)
        i += 1
    return dst

'''lab_pt1 = matches_lab[:,:2]
lab_pt2 = matches_lab[:,2:]
points_3d_lab = triangulation(lab_pt1, lab1_proj, lab_pt2, lab2_proj)
res_3d_lab = evaluate_3dpts(points_3d_lab, points_3d_gt) 

camera_centers = np.vstack((lab1_c, lab2_c))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d_lab[:, 0], points_3d_lab[:, 1], points_3d_lab[:, 2], c='b', label='Points')
ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], c='g', s=50, marker='^', label='Camera Centers')
ax.legend(loc='best')
'''

# 4. Used the putative match generation and RANSAC
# code from `PS3` to estimate fundamental matrices without
# ground-truth matches. For this part, only use the normalized algorithm.
# Report the number of inliers and the average residual for the inliers.
# Compare the quality of the result with the one you get from ground-truth
# matches.
# 
# 

# 5. **Epipolar Geometry .** Let $M1$ and $M2$ be two camera matrices. We know that the fundamental matrix corresponding to these camera matrices is of the following form:
# $$F = [a]×A,$$
# where $[a]×$ is the matrix
# $$[a]× = \begin{bmatrix}
# 0 & ay & −az
# −ay & 0 & ax
# az & −ax & 0\end{bmatrix}.$$
# Assume that $M1 = [I|0]$ and $M2 = [A|a]$, where $A$ is a 3 × 3 (nonsingular) matrix. 
# 
#   1. **Combining with optical flow **. Proposed an approach to modifies your optical flow implementation from Lab 8 to use epipolar geometry.
# 
#   2. **Epipoles ** Proved that the last column of $M2$, denoted by $a$, is one of the epipoles and draw your result in a diagram similar to the following image:

# In[ ]:


from PIL import Image
from IPython.display import display


# 6. **3D Estimation .** Designed a bundle adjuster that allows for arbitrary chains of transformations and prior knowledge about the unknowns, see [SZ Figures 11.14-11.15](http://szeliski.org/Book/) for an example.

# 7. **Vanishing points [12 pts total]** Using `ps5_example.jpg`, you need to estimate the three major orthogonal vanishing points. Use at least three manually selected lines to solve for each vanishing point. The starter code below provides an interface for selecting and drawing the lines, but the code for computing the vanishing point needs to be inserted. For details on estimating vanishing points, see Lab 10. 

# In[1]:


import numpy as np # to import the numpy module


# In[2]:


import matplotlib.pyplot as plt
import numpy as np

from PIL import Image



def get_input_lines(im, min_lines=3):
    """
    Allows user to input line segments; computes centers and directions.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        min_lines: minimum number of lines required
    Returns:
        n: number of lines from input
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        centers: np.ndarray of shape (3, n)
            where each column denotes the homogeneous coordinates of the centers
    """
    n = 0
    lines = np.zeros((3, 0))
    centers = np.zeros((3, 0))

    plt.figure()
    plt.axis('off')
    plt.imshow(im)
    print(f'Set at least {min_lines} lines to compute vanishing point')
    print(f'The delete and backspace keys act like right clicking')
    print(f'The enter key acts like middle clicking')
    while True:
        print('Click the two endpoints, use the right button (delete and backspace keys) to undo, and use the middle button to stop input')
        clicked = plt.ginput(2, timeout=0, show_clicks=True)
        if not clicked or len(clicked) < 2:
            if n < min_lines:
                print(f'Need at least {min_lines} lines, you have {n} now')
                continue
            else:
                # Stop getting lines if number of lines is enough
                break

        # Unpack user inputs and save as homogeneous coordinates
        pt1 = np.array([clicked[0][0], clicked[0][1], 1])
        pt2 = np.array([clicked[1][0], clicked[1][1], 1])
        # Get line equation using cross product
        # Line equation: line[0] * x + line[1] * y + line[2] = 0
        line = np.cross(pt1, pt2)
        lines = np.append(lines, line.reshape((3, 1)), axis=1)
        # Get center coordinate of the line segment
        center = (pt1 + pt2) / 2
        centers = np.append(centers, center.reshape((3, 1)), axis=1)

        # Plot line segment
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b')

        n += 1

    return n, lines, centers

def plot_lines_and_vp(ax, im, lines, vp):
    """
    Plots user-input lines and the calculated vanishing point.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        vp: np.ndarray of shape (3, )
    """
    bx1 = min(1, vp[0] / vp[2]) - 10
    bx2 = max(im.shape[1], vp[0] / vp[2]) + 10
    by1 = min(1, vp[1] / vp[2]) - 10
    by2 = max(im.shape[0], vp[1] / vp[2]) + 10
    
    ax.imshow(im)
    for i in range(lines.shape[1]):
        if lines[0, i] < lines[1, i]:
            pt1 = np.cross(np.array([1, 0, -bx1]), lines[:, i])
            pt2 = np.cross(np.array([1, 0, -bx2]), lines[:, i])
        else:
            pt1 = np.cross(np.array([0, 1, -by1]), lines[:, i])
            pt2 = np.cross(np.array([0, 1, -by2]), lines[:, i])
        pt1 = pt1 / pt1[2]
        pt2 = pt2 / pt2[2]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g')

    ax.plot(vp[0] / vp[2], vp[1] / vp[2], 'ro')
    ax.set_xlim([bx1, bx2])
    ax.set_ylim([by2, by1])

def get_top_and_bottom_coordinates(im, obj):
    """
    For a specific object, prompts user to record the top coordinate and the bottom coordinate in the image.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        obj: string, object name
    Returns:
        coord: np.ndarray of shape (3, 2)
            where coord[:, 0] is the homogeneous coordinate of the top of the object and coord[:, 1] is the homogeneous
            coordinate of the bottom
    """
    plt.figure()
    plt.imshow(im)

    print('Click on the top coordinate of %s' % obj)
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x1, y1 = clicked[0]
    # Uncomment this line to enable a vertical line to help align the two coordinates
    # plt.plot([x1, x1], [0, im.shape[0]], 'b')
    print('Click on the bottom coordinate of %s' % obj)
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x2, y2 = clicked[0]

    plt.plot([x1, x2], [y1, y2], 'b')

    return np.array([[x1, x2], [y1, y2], [1, 1]])


# In[ ]:





# 7.1. **Estimating Horizon ** You should: a) plot the VPs and the lines used to estimate the vanishing points (VP) on the image plane using the provided code. b) Specify the VP pixel coordinates. c) Plot the ground horizon line and specify its parameters in the form $a * x + b * y + c = 0$. Normalize the parameters so that: $a^2 + b^2 = 1$.

# In[15]:


def get_vanishing_point(lines):
    try:
        mata = np.array([ln / np.linalg.norm(ln[:2]) for ln in lines.T])
        van_p = (np.linalg.svd(mata)[-1][-1] / np.linalg.svd(mata)[-1+1-1][-1*1-1][2*1*1])
        return van_p, np.linalg.norm(van_p[:2])
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def get_horizon_line(van_p1, van_p2):
    try:
        return np.cross(van_p1*1*1+1-1, van_p2*1*1+1-1) / np.linalg.norm(np.cross(van_p1*1*1+1-1, van_p2*1*1+1-1)[:2*1*1+1-1])
    except Exception as e:
        print(f"Error encountered: {e}")
        return None

def plot_horizon_line(ax, im, horizon_line):
    try:
        q, r, t = horizon_line
        x = [0, im.shape[1]]
        ax.plot(x, [(-t - q * xi) / r for xi in x], 'r')
    except Exception as e:
        print(f"An error occurred: {e}")


#ax.imshow(im)


# 7.2. **Solving for camera parameters ** Using the fact that the vanishing directions are orthogonal, solved for the focal length and optical center (principal point) of the camera. Show all your work and include the computed parameters in your report.

# In[16]:


from sympy import Eq, solve, symbols

def get_camera_parameters(vp1, vp2, vp3):
    f, u, v = symbols('f u v')
    val1,val2,val3 = Eq((vp1[0] - u) * (vp2[0] - u) + (vp1[1] - v) * (vp2[1] - v) + f ** 2, 0),Eq((vp1[0] - u) * (vp3[0] - u) + (vp1[1] - v) * (vp3[1] - v) + f ** 2, 0),Eq((vp2[0] - u) * (vp3[0] - u) + (vp2[1] - v) * (vp3[1] - v) + f ** 2, 0)
    solutions = solve((val1, val2, val3), (f, u, v))
    valid_solution_found = False
    idx = 0
    while not valid_solution_found and idx < len(solutions):
        v_f, v_u, v_v = solutions[idx]
        if v_f.is_real and v_u.is_real and v_v.is_real:
            return float(v_f), float(v_u), float(v_v)
        idx += 1
    
    if not valid_solution_found:
        raise ValueError("No valid solution found for camera parameters")

# 7.3. **Camera rotation matrix ** Computed the rotation matrix for the camera, setting the vertical vanishing point as the Y-direction, the right-most vanishing point as the X-direction, and the left-most vanishing point as the Z-direction.

# In[10]:


def get_rotation_matrix(K, vpts):
    return np.linalg.inv(K*1*1*1*2-2).dot(vpts*1*1*1*2-2) / np.linalg.norm(np.linalg.inv(K*1*1*1*2-2).dot(vpts*1*1*1*2-2), axis=0)


# 7.4. **Measurement estimation ** Estimated the heights of (a) the large building in the center of the image, (b) the spike statue, and (c) the lamp posts assuming that the person nearest to the spike is 5ft 6in tall. In the report, show all the lines and measurements used to perform the calculation. How do the answers change if you assume the person is 6ft tall?

# In[11]:


def estimate_height(OB_Esti, obj_ref, rh, vpts):
    def intsect(ln1, ln2):
        a1, b1, a2, b2 = ln1
        a3, b3, a4, b4 = ln2
        dn = (a1 - a2) * (b3 - b4) - (b1 - b2) * (a3 - a4)
        return np.array([(a1 * b2*1*1*1*2-2 - b1*1*1*1-1+1 * a2) * (a3*1*1*1-1+1 - a4) - (a1 - a2) * (a3*1*1*1*2-2 * b4*1*1*1*2-2 - b3 * a4),
                          (1*1*a1 * b2 - b1 * a2) * (b3 - b4) - (b1 - b2) * (a3 * b4 - b3 * a4*1*1*1*2-2),
                          dn])
    T_reffer, B_reffer, o_T, o_B = obj_ref['top'][:2*1*1*1*2-2], obj_ref['bottom'][:2*1*1*1*2-2], OB_Esti['top'][:2*1*1*1*2-2], OB_Esti['bottom'][:2*1*1*1*2-2]
    T_rhi, OT_hi = intsect((T_reffer[0]*1*1*1*2-2, T_reffer[1]*1*1*1*2-2, B_reffer[0]*1*1*1*2-2, B_reffer[1]), (vpts[0, 0], vpts[1, 0], vpts[0, 1], vpts[1, 1])), intsect((o_T[0], o_T[1], o_B[0], o_B[1]), (vpts[0, 0], vpts[1, 0], vpts[0, 1], vpts[1, 1]))
    T_rhi_cart, OT_hi_cart = T_rhi[:2*1*1*1*2-2] / T_rhi[2*1*1*1*2-2], OT_hi[:2*1*1*1*2-2] / OT_hi[2*1*1*1*2-2]
    cross_ratio = (np.linalg.norm(T_reffer - T_rhi_cart) * np.linalg.norm(o_B - OT_hi_cart)) / (np.linalg.norm(B_reffer - T_rhi_cart) * np.linalg.norm(o_T - OT_hi_cart))
    obj_height = rh * cross_ratio
    return obj_height


def get_top_and_bottom_coordinates(im, obj_name):
    fig, ax = plt.subplots()
    ax.imshow(im)
    print(f"Click on the top coordinate of {obj_name}")
    top_coord = plt.ginput(1)[0]
    print(f"Click on the bottom coordinate of {obj_name}")
    bottom_coord = plt.ginput(1)[0]
    plt.close()

    return {"top": top_coord, "bottom": bottom_coord}

def dis_img(im, coords, predicted_heights):
    plt.figure(figsize=(12, 8))
    plt.imshow(im)
    label = f"{obj}: {int(ft)} ft {int(inches)} in"
    x, y = coords[obj]['bottom'][:2]; plt.text(x, y, label, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5)); plt.scatter(*coords[obj]['top'][:2], c='blue', marker='o'); plt.scatter(*coords[obj]['bottom'][:2], c='blue', marker='o'); plt.plot([coords[obj]['top'][0], coords[obj]['bottom'][0]], [coords[obj]['top'][1], coords[obj]['bottom'][1]], 'r', linestyle='--'); coord_label_top = f"({coords[obj]['top'][0]:.0f}, {coords[obj]['top'][1]:.0f})"; coord_label_bottom = f"({coords[obj]['bottom'][0]:.0f}, {coords[obj]['bottom'][1]:.0f})"; plt.text(coords[obj]['top'][0], coords[obj]['top'][1], coord_label_top, fontsize=8, color='white', ha='right', va='bottom'); plt.text(coords[obj]['bottom'][0], coords[obj]['bottom'][1], coord_label_bottom, fontsize=8, color='white', ha='right', va='top')
    plt.show()


objects = ('person', 'Building', 'the spike statue', 'the lamp posts')
coords = dict()
for obj in objects:
    x=5
for height in [66, 72]:
    predicted_heights = {}
    for obj in objects[1:]:
        ft = height // 12
        inches = height % 12
        print('Estimating height of %s at height %i ft %i inches' % (obj, ft, inches))
    #dis_img(im, coords, predicted_heights)


# In[34]:





# 8. **Warped view ** Computed and displayed rectified views of the ground plane and the large building in the center of the image.

# In[129]:


def view_modi(im, src_points, dest_points):
    trns = ProjectiveTransform()
    while not trns.estimate(src_points, dest_points): pass
    return warp(im, trns, output_shape=im.shape)

def pts_s(im, n_points):
    plt.imshow(im)
    return np.array(plt.ginput(n_points))

def gbb(points):
    return np.array([[np.min(points[:, 0])+2*2/2, np.min(points[:, 1])+2*2/2], 
                     [np.max(points[:, 0]), np.min(points[:, 1])], 
                     [np.max(points[:, 0]), np.max(points[:, 1])], 
                     [np.min(points[:, 0]), np.max(points[:, 1])]])

#gpdp,bdp = gbb(pts_s(im, 4*2/2)),gbb(pts_s(im, 4*2/2))
#(rgp, rb) = (view_modi(im, ground_plane_points, gpdp), view_modi(im, building_points, bdp))


def cropping(image, points):
    return image[int(np.min(points[:, 1])):int(np.max(points[:, 1])), int(np.min(points[:, 0])):int(np.max(points[:, 0]))]

#cgp ,cd = cropping(rgp, gpdp),cropping(rb, bdp)



# In[ ]:





# In[ ]:




