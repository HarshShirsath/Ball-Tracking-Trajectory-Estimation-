
# Perception for Autonomous Robots
### Trajectory of a Thrown Ball
#### Instructions:
1. **Run the problem1.py script.**
2. The script reads a video file (`ball.mp4`) and filters the red channel to detect and plot the pixel coordinates of the center point of the ball.
3. Utilizes Standard Least Squares to fit a curve to the extracted coordinates.
   - Prints the equation of the curve.
   - Plots the data with the best fit curve.
4. Computes the x-coordinate of the ball’s landing spot based on the specified conditions.

### LIDAR Point Cloud Data Analysis
#### Instructions:
1. **Run the problem2.py script.**
2. Computes the covariance matrix and surface normal for the given LIDAR point cloud data in `pc1.csv`.
3. Implements Standard Least Squares, Total Least Squares, and RANSAC to fit a surface to the data.
   - Plots the results for each method and provides an interpretation.
   - Discusses the steps and parameters used in the RANSAC implementation.
4. Compares and discusses the suitability of outlier rejection methods.

### Additional Notes:
- Ensure that the required dependencies (OpenCV, Matplotlib) are installed.
- For any questions or issues, please contact the contributors.
- Enjoy exploring the fascinating world of perception for autonomous robots!
