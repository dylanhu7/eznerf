# Rays
One of the first steps in NeRF is to generate rays through the scene from the virtual camera.

We then sample along these rays to generate a set of points and directions in space to feed into our neural network.

## Computer Graphics Prerequisites
We borrow from the traditional computer graphics idea of ray casting to generate rays through the scene.

As such, it may be helpful to cover some computer graphics prerequisites.

### The Pinhole Camera Model
We imagine our virtual camera as a single point in space and having and infinitesimally small aperture, allowing for convenient mathematical properties.

<figure>

![Pinhole Camera Model](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Pinhole-camera.svg/2880px-Pinhole-camera.svg.png)
<figcaption style="text-align: center; color: GrayText; font-size: 0.9em;">
    <i>The pinhole camera model</i>
    <a href="https://en.wikipedia.org/wiki/Pinhole_camera_model">from Wikipedia</a>
</figcaption>

</figure>

In the above diagram, notice how the red "rays" travel from the tree to the pinhole. In real life, these rays extend through the pinhole onto the image plane (camera sensor) and produce the inverted image. However, in computer graphics, we generally do not consider the image plane and instead model an imaginary view plane (discussed next!)

In reality, "rays" of light enter a camera or your eye from all directions, but in computer graphics we generally think of rays being shot *from* the camera and into the scene.

### The Pose Matrix
The pose matrix is a 4x4 matrix that describes the position and orientation of the camera in the scene. It is sometimes also referred to as the "transform" matrix.

A useful way to think about the pose matrix is a *transformation* that takes the origin of the world coordinate system and moves it to the position of the camera and rotates it to the orientation of the camera.

We can understand *why* we can think of the pose matrix this transformation by noticing that it is simply the composition of a translation matrix and a rotation matrix.

#### Translation Matrix
A translation matrix is a transformation which takes a point in space and moves it by a certain amount in the $x$, $y$, and $z$ directions.

The 4x4 translation matrix takes the form:

$$\begin{bmatrix} 1 & 0 & 0 & x \\ 0 & 1 & 0 & y \\ 0 & 0 & 1 & z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
where $x$, $y$, and $z$ are the amounts to translate in the $x$, $y$, and $z$ directions, respectively.

As an example to demonstrate that this works, we can translate the point $(1, 2, 3)$ by $(-2, 4, 3)$ to get $(-1, 6, 6)$.

The point $(1, 2, 3)$ is represented as a 4D column vector in homogeneous coordinates (the last coordinate is 1 because it is a point; if it were a vector, the last coordinate would be 0):

$$\begin{bmatrix} 1 \\ 2 \\ 3 \\ 1 \end{bmatrix}$$

Multiplying this vector by the translation matrix gives us the following:

$$\begin{bmatrix} 1 & 0 & 0 & -2 \\ 0 & 1 & 0 & 4 \\ 0 & 0 & 1 & 3 \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 3 \\ 1 \end{bmatrix} = \begin{bmatrix} -1 \\ 6 \\ 6 \\ 1 \end{bmatrix}$$

#### Rotation Matrix
A rotation matrix is a transformation which takes a point in space and rotates it around the $x$, $y$, and $z$ axes.

Importantly, we generally rotate *first* before translating, as rotating a point not at the origin around an axis will cause the point to move.

The 4x4 rotation matrix for rotations along all three axes is the composition of three 4x4 rotation matrices, one for each axis. The rotation matrix for a rotation around the $x$ axis is:

$$M_x = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos(\theta_x) & -\sin(\theta_x) & 0 \\ 0 & \sin(\theta_x) & \cos(\theta_x) & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

where $\theta$ is the angle in radians to rotate around the $x$ axis.

The rotation matrices for the $y$ and $z$ axes are similar:

$$M_y = \begin{bmatrix} \cos(\theta_y) & 0 & \sin(\theta_y) & 0 \\ 0 & 1 & 0 & 0 \\ -\sin(\theta_y) & 0 & \cos(\theta_y) & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad M_z = \begin{bmatrix} \cos(\theta_z) & -\sin(\theta_z) & 0 & 0 \\ \sin(\theta_z) & \cos(\theta_z) & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

To illustrate that this works, we can rotate the direction vector $(0, 1, 0)$ by $\pi/2$ radians around the $x$ axis to get $(0, 0, 1)$.

$$\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos(\pi/2) & -\sin(\pi/2) & 0 \\ 0 & \sin(\pi/2) & \cos(\pi/2) & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}$$

#### Putting it Together
We can now put the translation and rotation matrices together to form the pose matrix. In general, to compose two linear transformations, we simply multiply their matrices together.

In this case, using a rotation by $\pi/2$ around the $x$ axis and a translation by $(1, 2, 3)$, we can derive a transformation which first rotates and then translates as such by performing the matrix multiplication of the corresponding 4x4 matrices:

$$\begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 0 & 2 \\ 0 & 0 & 1 & 3 \\ 0 & 0 & 0 & 1 \end{bmatrix}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos(\pi/2) & -\sin(\pi/2) & 0 \\ 0 & \sin(\pi/2) & \cos(\pi/2) & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & \cos(\pi/2) & -\sin(\pi/2) & 2 \\ 0 & \sin(\pi/2) & \cos(\pi/2) & 3 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

We see that in general, we form the 4x4 pose matrix by composing the upper 3x3 rotation matrix with the upper right 3x1 translation column vector, along with a bottom row of $(0, 0, 0, 1)$.

In the code, we extract the rotation and translation components of the pose matrix and use them to transform our ray direction vectors.


### The View Plane
The view plane is an imaginary plane perpendicular to the direction the camera is looking. We use this convenient abstraction to calculate the directions of the rays we will shoot into the scene.

<figure>

![Pinhole Camera Model](https://cs1230.graphics/projects/ray/1/viewplane.png)
<figcaption style="text-align: center; color: GrayText; font-size: 0.9em;">
    <i>The view plane</i>
    <a href="http://cs1230.graphics/projects/ray/1-algo-ans">from Brown CS1230</a>
</figcaption>

</figure>

We place the view plane an arbitrary distance $k$ from the camera. In the code, we use $k = 1$, so $k$ is never explicitly referenced.

The view plane is subdivided according to the dimensions of the image we are rendering. For instance, if we are rendering a 640x480 image, the view plane is subdivided into 640 columns and 480 rows.

> Note: the view plane is similar to the real image plane mentioned above, as the image is projected onto the view plane. However, the view plane lies between our camera and the scene, and so the image is not upside-down on our view plane.

> Note: in 2D contexts of computer graphics (such as when discussing the view plane), $x$ and $y$ are used to denote the horizontal and vertical axes, respectively. $(0, 0)$ is located at the top-left of the view plane, and positive $x$ points to the right and positive $y$ points down (as in the diagram above).

## Ray Generation
We now have all the pieces we need to generate rays through the scene.

We can generate rays in the following way:
1. Determine the position of each pixel center on the view plane.
