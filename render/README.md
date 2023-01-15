# Render
After passing our samples (stratified, hierarchical, or both) through the encoder and MLP and receiving the corresponding predictions for color $\mathbf{c}$ and volume density $\sigma$, we can volume render our scene to an image.

## Expected Color of a Ray
> Reminder: we express a ray as a parametric function of $t$:
> $$ \mathbf{r}(t) = \mathbf{o} + t\mathbf{d} $$
> where $\mathbf{r}$ is the position along the ray, $\mathbf{o}$ is the ray origin, and $\mathbf{d}$ is the ray direction.

The expected color of a ray is computed as the following, potentially intimidating integral:
$$\begin{equation} C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt\text{, where }T(t) = \exp\left(-\int_{t_n}^t\sigma(\mathbf{r}(s))\,ds\right)\end{equation}$$

Expressing the product $T(t)\sigma(\mathbf{r}(t))$ as a single weight term $w(t)$, we can simplify the integral to:
$$\begin{equation} C(\mathbf{r}) = \int_{t_n}^{t_f} w(t)\,\mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt\text{, where }w(t) = T(t)\sigma(\mathbf{r}(t))\end{equation}$$

Despite all the notation, the intuition is actually fairly easy to understand!

We can break down each team and its definition and role in the integral.

### Color
$\mathbf{c}(\mathbf{r}(t), \mathbf{d})$ is simply the RGB color of the point $\mathbf{r}(t)$ along the ray predicted by the MLP. Note that $\mathbf{c}$ is a function of both position and direction - this helps gives NeRF view dependence!

### Volume Density
$\sigma(\mathbf{r}(t))$ is the volume density of the point $\mathbf{r}(t)$ along the ray predicted by the MLP. It can be interpreted as "the differential probability of a ray terminating at an infinitesimal particle at location $\mathbf{r}(t)$". In other words, it is the probability that the ray will be occluded by a surface at this point along the ray.

### Transmittance
For a point $\mathbf{r}(t)$ along a ray, we want to add its contribution to the expected color of the ray. However, we don't want to add the contribution of a point if it is occluded by a surface closer to the camera.

To account for occlusion, we add a transmittance term $T(t)$ representing the probability that a ray will not be occluded by a surface between $t_n$ and $t$ (between the start of where we begin sampling to where the current point is).

$T(t)$ is computed as the exponential of the negative integral of the volume density along the ray up to the current point. This is because the volume density $\sigma(\mathbf{r}(s))$ is the probability that a ray *will* be occluded by a surface at a given point along the ray. We take the exponential of the negative accumulation of volume density, as this forces $T$ values to be between 0 and 1 while allowing for the negative integral to take on values in $(-\infty, 0]$. As such, if there is no occlusion, $T(t) = e^{0} = 1$, and if there is full occlusion, $T(t) = e^{-\infty} = 0$.