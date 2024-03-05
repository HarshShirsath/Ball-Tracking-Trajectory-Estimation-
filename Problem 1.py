import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import matrix

#1.1-----------------------------------------------------------------------------------------------------------------------------  
cap = cv2.VideoCapture('F:/Perception(673)/Projects/Project 1/shirsath_proj1/ball.mp4')
x_array=[]
y_array=[]

while(1):
    ret, frame = cap.read()
    if ret == True:
    # It converts the BGR color space of image to HSV color space
            red = np.uint8([[[255,0,0 ]]])
            hsv_red = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
      
    # Threshold of red in HSV space
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([2, 255, 255])
  
    # preparing the mask to overlay
            mask = cv2.inRange(hsv_red, lower_red, upper_red)
        # red = np.uint8([[[255,0,255]]])
    
    # Display the resulting frame
            cv2.imshow("red",frame)
           #print('vi')
 
    # Press Q on keyboard to  exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                     break
    else:
            break

      
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-red regions
    result = cv2.bitwise_and(frame, frame, mask = mask)
  
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)


    non_zero= np.where(mask>0) #2d array)
#     print(non_zero)
#     pixelpoints = cv.findNonZero(mask)

# Extracting the non-zero values from the array especially rows non_zero[]
    y=non_zero[0]
    # print("x=",y)
    x=non_zero[1]
#     print("y=",x)

# Calculating the mean of the function 
    if(len(x)>0 and len(y)>0):
        x_mean=np.mean(x)
        y_mean= np.mean(y)
        # print('x_mean:', x_mean)
        # print('y_mean', y_mean)
        #removing all the nan values from the arrays of mean
        if not (math.isnan(x_mean) and math.isnan(y_mean) ):
                x_array.append(x_mean)
                y_array.append(y_mean)
# print(x_array)
# print(y_array)


# 1.2------------------------------------------------------------------------------------------------------------------

# Finding the Least Square Line Fitting

# B=(X.T*X)**-1(X.T*Y)
#Converting lists to arrays
x_array=np.array(x_array)
y_array= np.array(y_array)
A = np.column_stack([x_array**2, x_array, np.ones(len(x_array))]) #X matrix
# print('A:',np.shape(A))

#Finding the transpose of the matrix A
A_t= np.transpose(A)                                              #X.T= A transpose matrix
# print('A_t:',A_t)                                               #Printing X.T= A transpose matrix
# print(np.size(A))

#Dot product of X.T*X
B_A=np.dot(A_t,A)                                                 #X.T*X
# print(np.shape(B_A), B_A)
# B_inv= np.linalg.inv(B_A)
B_a= np.dot(A_t,y_array)                                          #X.T*Y          
B= np.dot(B_A,B_a)                                                #(X.T*X)**-1(X.T*Y)
# print('B:',np.shape(B))
# print('y_array:',np.shape(y_array))

# print('B:',B)
# Extracting the coefficient of B[] matrix 
a= B[0]            
b= B[1]
c= B[2]
# print('B_x', B_x)
# print('B_y', B_y)
# print('B_z', B_z)

#Define the range of the x coordinates inorder to get the y coordinates
x_range=np.linspace(min(x_array),max(x_array), 100)
# print('x_range:', x_range)
x_range=np.array(x_range)                               # Converting list to array 
# print('x_range', x_range)


# Matrix B and it's elements
coefficient= np.linalg.solve(B_A,B_a)
a,b,c = coefficient                                     # Coefficients of the matrix B[]
y_range = a*x_range**2 + (b*x_range) + c
print(f'y = {a}*x**2 + {b}*x + {c}', y_range)           #print the parabolic equations 

plt.figure()
plt.gca().invert_yaxis()
plt.plot(x_array, y_array , 'bo', "blue" )              #Plotting the x_array & y_array which is the center of mean co-ordinates of x and y
# plt.gca().invert_yaxis()
plt.plot(x_range, y_range,'-', "red" )                  #Least Square Fitting curve fir plot
plt.show()

# 1.3----------------------------------------------------------------------------------------------------------------------------------
# X-coordinate of the ball’s landing spot in pixels
landing_y = 300 + y_array[0]                                    #Landing y-Coordinates pixels
landing_x = (-b + np.sqrt(b**2 - 4*a*(c - landing_y)))/(2*a)    #Landing x-Coordinates pixels
print(f"Landing x-coordinate:" ,{landing_x} )

cv2.destroyAllWindows()
cap.release()