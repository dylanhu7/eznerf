# Render
After passing our samples (stratified, hierarchical, or both) through the encoder and MLP and receiving the corresponding predictions for color $\mathbf{c}$ and volume density $\sigma$, we can volume render our scene to an image.

## Expected Color of a Ray
> Reminder: we express a ray as a parametric function of $t$:
> ```math
>  \mathbf{r}(t) = \mathbf{o} + t\mathbf{d}
> ```
> where $\mathbf{r}$ is the position along the ray, $\mathbf{o}$ is the ray origin, and $\mathbf{d}$ is the ray direction.

### Continuous Expression
The expected color of a ray is computed as the following, potentially intimidating integral:
```math
\begin{equation} C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt\text{, where }T(t) = \exp\left(-\int_{t_n}^t\sigma(\mathbf{r}(s))\,ds\right)\end{equation}
```

### Discrete Expression
In our discrete case with $N$ samples $(\mathbf{c}\_i, \delta_i)$ where $\delta_i = t_{i + 1} - t_i$, we can approximate this integral as:
```math
\begin{equation} \hat{C}(\mathbf{r}) = \sum_{i=1}^N T_i\,(1 - \exp(\sigma_i\delta_i))\,\mathbf{c}_i \text{, where }T_i = \exp\left(-\sum_{j = 1}^{i - 1}\sigma_j\delta_j\right)\end{equation}
```

> An intuition for why we go from $\sigma(\mathbf{r}(t))$ in the continuous case to $(1 - \exp(\sigma_i\delta_i))$ in the discrete case is that $T_i$ only captures transmittance until $t_i$ while we are interested in the contribution of the spatial interval $\delta_i$ from $t_i$ to $t_{i + 1}$. Our corresponding term for volume density is therefore dependent on $\delta_i$, and we can think of it as the probability that a ray will be occluded by a surface in the interval $\delta_i$ (the opposite of [transmittance](#transmittance)); note that $(1 - \exp(\sigma_i\delta_i))$ takes a similar form to $T_i$ itself.

#### Discrete Expression with Weight Term
Expressing the product $T_i(1 - \exp(\sigma_i\delta_i))$ as a single weight term $w_i$, we can rewrite as:
```math
\begin{equation} \hat{C}(\mathbf{r}) = \sum_{i = 1}^N w_i\mathbf{c}_i\text{, where }w_i = T_i\,(1 - \exp(\sigma_i\delta_i))\end{equation}
```

Despite all the notation, the intuition is actually fairly easy to understand!

We can break down each team and its definition and role in the integral.

### Breaking it Down
#### Color
$\mathbf{c}(\mathbf{r}(t), \mathbf{d})$ is simply the RGB color of the point $\mathbf{r}(t)$ along the ray predicted by the MLP. Note that $\mathbf{c}$ is a function of both position and direction - this helps gives NeRF view dependence!

#### Volume Density
$\sigma(\mathbf{r}(t))$ is the volume density of the point $\mathbf{r}(t)$ along the ray predicted by the MLP. It can be interpreted as "the differential probability of a ray terminating at an infinitesimal particle at location $\mathbf{r}(t)$". In other words, it is the probability that the ray will be occluded by a surface at this point along the ray.

#### Transmittance
For a point $\mathbf{r}(t)$ along a ray, we want to add its contribution to the expected color of the ray. However, we don't want to add the contribution of a point if it is occluded by a surface closer to the camera.

To account for occlusion, we add a transmittance term $T(t)$ representing the probability that a ray will not be occluded by a surface between $t_n$ and $t$ (between the start of where we begin sampling to where the current point is).

$T(t)$ is computed as the exponential of the negative integral of the volume density along the ray up to the current point. This is because the volume density $\sigma(\mathbf{r}(s))$ is the probability that a ray *will* be occluded by a surface at a given point along the ray. We take the exponential of the negative accumulation of volume density, as this forces $T$ values to be between 0 and 1 while allowing for the negative integral to take on values in $(-\infty, 0]$. As such, if there is no occlusion, $T(t) = e^{0} = 1$, and if there is full occlusion, $T(t) = e^{-\infty} = 0$.

#### Weight
Recall from [equation 3](#discrete-expression-with-weight-term) that the weight term $w_i$ is simply the product of the transmittance and volume density.

The intuition behind why we refer to this term as the weight is that it directly modulates how much of the color $\mathbf{c}_i$ is added to the expected color of the ray.

We use these weights to generate a probability distribution in [hierarchical sampling](../sample/README.md#hierarchical-sampling).

## Implementation: `volume_render`
| Parameter | Type | Description |
| --- | --- | --- |
| `color` | `Tensor` | A `[image_height, image_width, num_samples, 3]` tensor of RGB color predictions from the MLP |
| `sigma` | `Tensor` | A `[image_height, image_width, num_samples]` tensor of volume density predictions from the MLP |
| `t` | `Tensor` | A `[num_samples]` tensor of $t$ values for each sample along the ray |

First, we compute `num_samples - 1` $\delta$ values by subtracting each $t$ value from the next:
```py
deltas = t_vals[..., 1:] - t_vals[..., :-1]
```

We then omit color and volume density values for the last sample, as we don't have a corresponding $\delta$ value for it:
```py
color = color[..., :-1, :]
sigma = sigma[..., :-1]
```

We compute $\alpha_i = 1 - \exp(\sigma_i\delta_i)$ for use in computing transmittance and weights:
```py
alphas = 1. - torch.exp(-sigma * deltas)
```

In order to compute transmittance, we need to compute the cumulative product of $\alpha_i$ along the ray. PyTorch's [`torch.cumprod`](https://pytorch.org/docs/stable/generated/torch.cumprod.html) will get us mostly there, but we write a custom variation of it to make it "exclusive". This means that the first element of the cumulative product is 1, and the last element is the product of all elements in the input tensor except the last. This is useful for computing transmittance, as we want the first element to be 1, and the last element to be the product of all $\alpha$ values before the current $\delta$.
```py
def cumprod_exclusive(tensor: Tensor) -> Tensor:
    """An exclusive cumulative product"""
    # [image_height, image_width, num_samples]
    return torch.cumprod(torch.cat(
        [torch.ones_like(tensor[..., :1]), tensor[..., :-1]], dim=-1), dim=-1)
```

We can now compute the transmittance $T_i$ for each $\delta$:
```py
transmittance = cumprod_exclusive(1. - alphas)
```

Then we compute the weight $w_i$ for $\delta_i$ as the product of $T_i$ and $\alpha_i$:
```py
weights = transmittance * alphas
```

Finally, we compute the expected color of the ray by summing the product of the color and weight for each $\delta$:
```py
image = torch.sum(weights[..., None] * color, dim=-2)
```

Along with the rendered image, we also return the weights and deltas for use in [hierarchical sampling](../sample/README.md#hierarchical-sampling).