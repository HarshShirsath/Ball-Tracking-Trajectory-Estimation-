import numpy as np
import matplotlib.pyplot as plt
# To fit a plane to a set points using Standard Least Squares a function is defined below

def stnd_least_sq(extract_points):
    #  mean of the x, y, and z values from the list provided
    x_mean = np.mean(extract_points[:, 0])
    y_mean = np.mean(extract_points[:, 1])
    z_mean = np.mean(extract_points[:, 2])

    # Subtract the mean values from  x, y, and z list values provided
    centered_points = extract_points - np.array([x_mean, y_mean, z_mean])
    # print('centered_points:', centered_points)

    # Compute the covariance matrix of the centered points
    covariance_mat = np.cov(centered_points.T)
    # print('centered_points.T:', centered_points.T)

    # In the below line of code eigenvectors and eigenvalues of the covariance matrix are computed
    eigen_values, eigen_vectors = np.linalg.eig(covariance_mat)
    # print('eigen_values, eigen_vectors:', eigen_values, eigen_vectors)

    # The normal vector is the eigenvector corresponding to the smallest eigenvalue
    normal = eigen_vectors[:, np.argmin(eigen_values)]
    magnitude = np.linalg.norm(normal)
    direction = np.arctan2(normal[1], normal[0])
    print('magnitude:', magnitude)
    print('direction:', direction)
    # print('normal:', normal)
    # print('Magnitude:', eigen_values)
    # A point on the plane can be found by using the mean of the x, y, and z values
    point = np.array([x_mean, y_mean, z_mean])
    print('point:', point)
    return normal, point


# To fit a plane to a set points using Total Least Squares a function is defined below

def total_least_square(extract_points):
    # Calculating the mean of points by defining a variable mean_points
    mean_points = np.mean(extract_points, axis=0)
    print('mean_points:', mean_points)

    #find the center points by subtracting the extracted point from the list with the mean of points from the column
    centre_points = extract_points - mean_points
    print('centre_points:', centre_points)

    # Singular Value Decomposition calculation for all the centered points
    A, B, S = np.linalg.svd(centre_points)
    #print('A, B, S:',A, B, S)

    # Last column of VT i.e -1 is the normal vector
    normal_line = S[-1, :]
    print('normal_line',normal_line)
 
    point = mean_points

    return normal_line, point


# Function to calculate the distance from a point normal to a plane
def pt_pt_dist(point, normal, plane_point):
    return np.abs(np.dot(normal, point - plane_point)) / np.linalg.norm(normal)

# Defining a function to fit a plane using RANSAC with the given set of data
def ransac(extract_points, iterations=800, inlier_threshold=0.1):
    normal_best_pt = None
    point_best = None
    inlier_maxim = 0

    # Reiterating the fitting process multiple times
    for i in range(iterations):
        # Selecting random 3 points from the  given data
        sampling_indices = np.random.choice(len(extract_points), size=3, replace=False)
        sampling_points = extract_points[sampling_indices]
        # print('sampling_indices:',sampling_indices)
        # print('sampling_points:',sampling_points)

        # Plane Fitting to the sample points using Standard Least Squares
        normal, point = stnd_least_sq(sampling_points)
        # print('normal, point:',normal, point)
        
        # Distance from each point to the fitted plane will be as
        distances = [pt_pt_dist(point, normal, p) for p in extract_points]
        # print('distances:',distances)

        # Find the number of inliers within the threshold
        inlier_count = sum([1 for d in distances if d < inlier_threshold])
        # print('inlier_count:',inlier_count)
    
        #If the current inliers have more inliers then update the best fit plane
        if inlier_count > inlier_maxim:
            normal_best_pt = normal
            point_best = point
            inlier_maxim = inlier_count

    #  Best-fitting plane is returned
    return normal_best_pt, point_best

#  A function to plot a plane and points
def plot_surface(extract_points, normal_pt, point, title_plt):

    # Compute the distance from the plane to the origin
    d = -np.dot(normal_pt.T, point)
    print('d:',d)
    # Normalizing the normal vector
    norm = np.linalg.norm(normal_pt)
    unit_normal = normal_pt / norm
    print('unit_normal:',unit_normal)

    # Calculate the limits of the data in the x and y dimensions
    x_min, x_max = np.min(extract_points[:, 0]), np.max(extract_points[:, 0])
    y_min, y_max = np.min(extract_points[:, 1]), np.max(extract_points[:, 1])
    print('x_min, x_max:',x_min, x_max)
    print('y_min, y_max:',y_min, y_max)




    # Mesh Grid for the x and y dimensions to define a plane grid
    x__x, y__y = np.meshgrid(np.linspace(x_min, x_max, 10),
                         np.linspace(y_min, y_max, 10))
    z = (-unit_normal[0] * x__x - unit_normal[1] * y__y - d) * 1. / unit_normal[2]
    print('z:',z)

    # Plot the points and the plane
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    ax.set_title(title_plt)
    ax.scatter(extract_points[:, 0], extract_points[:, 1], extract_points[:, 2])
    ax.plot_surface(x__x, y__y, z, alpha=0.5)
    plt.show()


def main():
    # Loading the data from the file
    points_all = np.loadtxt('F:\Perception(673)\Projects\Project 1\Project 2.1\pc1.csv', delimiter=',')
    # print(points.shape)
    # Using Standard Least Squares: Fit a plane
    normal_standard_least_sqr, standard_least_sqr_list = stnd_least_sq(points_all)
    plot_surface(points_all, normal_standard_least_sqr, standard_least_sqr_list, "pc1_csv: Least Square Fitting")

    # Using Total Least Squares: Fit a plane
    normal_total_least_sqr, point_total_least_sqr = total_least_square(points_all)
    plot_surface(points_all, normal_total_least_sqr, point_total_least_sqr,
                 "pc1_csv:Total Least Square Fitting")

    # Using RANSAC: Fit a plane
    normal_ransac, point_ransac = ransac(points_all)
    plot_surface(points_all, normal_ransac, point_ransac, "pc1_csv:RANSAC fitting ")

    # From the file load the data 
    points_all = np.loadtxt('F:\Perception(673)\Projects\Project 1\Project 2.1\pc1.csv', delimiter=',')
    # print(points.shape)
    #  Using Standard Least Squares: Fit a plane 
    normal_standard_least_sqr, standard_least_sqr_list = stnd_least_sq(points_all)
    plot_surface(points_all, normal_standard_least_sqr, standard_least_sqr_list, "pc2_csv:Least Square Fitting")

    # Using Total Least Squares: Fit a plane 
    normal_total_least_sqr, point_total_least_sqr = total_least_square(points_all)
    plot_surface(points_all, normal_total_least_sqr, point_total_least_sqr,
                 "pc2_csv:Total Least Square Fitting")

    # Using RANSAC: Fit a plane
    normal_ransac, point_ransac = ransac(points_all)
    plot_surface(points_all, normal_ransac, point_ransac, "pc2_csv: RANSAC fitting")




if __name__ == "__main__":
    main()