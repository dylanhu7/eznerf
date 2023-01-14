# Sample
After we have generated our rays, we need to sample along them to accumulate points in space that we can pass into our neural network.

We employ two different sampling methods, stratified and hierarchical, at different stages of the pipeline. The stratified sampling method is used to generate an initial set of sample points along the rays, while hierarchical sampling is used after the neural network has evaluated the first set and returned a set of corresponding weights.

While stratified sampling converges to being uniformly random along the ray, hierarchical sampling allocates more samples to regions of space that are more likely to contribute to the final rendering.

## Stratified Sampling
The stratifed sampling approach is taken from the original paper, where we divide the space between the near and far planes into even strata, and we sample one point randomly within each stratum.

Over the course of the optimization, we will be able to converge to a uniform distribution of points along the ray.

We use stratified random sampling rather than simply sampling uniformly along the ray because it reduces the variance of the samples, which can benefit the optimization process, especially at the beginning.

### Implementation: `sample_stratified`
| Parameter | Type | Description |
| --- | --- | --- |
| `rays` | `Tensor` | A tensor of shape `[num_rays, image_width, image_height, 3]` containing the origins and directions of the rays. |
| `near` | `float` | The near plane of the camera. |
| `far` | `float` | The far plane of the camera. |
| `num_samples` | `int` | The number of samples to take along each ray. |

We first break down the rays tensor into the origins (which are all the camera position) and the directions:
```py
rays_o, rays_d = rays[..., 0], rays[..., 1]
```

Next, we compute the width of each stratum as the distance between the near and far planes divided by the number of samples:
```py
stratum_width = (far - near) / num_samples
```

We then use [`torch.linspace`](https://pytorch.org/docs/stable/generated/torch.linspace.html) to generate a tensor of shape `[num_samples]` which represent the beginning of each stratum:
```py
strata = torch.linspace(near, far - stratum_width, num_samples)
```

> Note: the values of `strata` are $t$ values along the ray, not world space coordinates. We express a position along a ray as a parametric function of $t$:
> $$ \mathbf{r}(t) = \mathbf{o} + t\mathbf{d} $$
> where $\mathbf{r}$ is the position along the ray, $\mathbf{o}$ is the ray origin, $\mathbf{d}$ is the ray direction, and $t$ is the parametric value.
>
> Since in our case the ray directions are not normalized but proportional to the distance from the camera to the corresponding point on the view plane, we can use the same $t$ values for all rays and span the entire volume.

We then use [`torch.rand_like`](https://pytorch.org/docs/stable/generated/torch.rand_like.html) to generate a tensor of shape `[num_samples]` which represent the offset of each sample within each stratum. We then add this offset to the beginning of each stratum to get the $t$ values of each sample:
```py
t = strata + torch.rand_like(strata) * stratum_width
```

Finally, we compute the position of each sample along the ray by multiplying the direction of the ray by the $t$ value and adding the origin:
```py
return rays_o[..., None, :] + t[..., None] * rays_d[..., None, :], t
```

> We also return the $t$ values of each sample for use in volume rendering and hierarchical sampling.

## Hierarchical Sampling
We recommend reading the [`render.py` README](../render/README.md) before reading this section, as we must volume render the scene using the stratified samples to generate corresponding weights for use in hierarchical sampling.

The hierarchical sampling approach is taken from the original paper as well. At a high level, we use the weights returned by the neural network to generate a probability distribution over the stratified samples, and we sample from this distribution to generate a new set of samples.

### Inverse Transform Sampling
In order to sample from a discrete (or continuous) probability distribution, we can use the [inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling) method. This method works by first generating a uniform random number between 0 and 1, and then finding the corresponding value in the cumulative distribution function (CDF) of the probability distribution.
#### Probability Density Function
As you may already be aware, the probability density function (PDF) of a random variable $X$ is the function that gives the probability that $X$ will take on a given value $x$:
$$f(x) = P(X = x)$$
In our case, we have a discrete probability distribution over a set of samples $x_i$:
$$p(x_i) = P(X = x_i)$$
> In the discrete case, we refer to the density function as the probability mass function (PMF).

<figure>
    <img src="https://i.imgur.com/w8y5SjS.png" alt="Probability Density Function">
    <figcaption>
        <p align="center">
            <i>A probability density function</i>
            <a href="https://www.geo.fu-berlin.de/en/v/soga/Basics-of-statistics/Continous-Random-Variables/The-Probability-Density-Function/index.html">from Freie Universität Berlin</a>
        </p>
    </figcaption>
</figure>

#### Cumulative Distribution Function
The cumulative distribution function (CDF) of a random variable $X$ is the function that gives the probability that $X$ will take on a value less than or equal to $x$:
$$F(x) = P(X \leq x)$$
It can also be represented as the integral of the PDF:
$$F(x) = \int_{-\infty}^x f(x) dx$$
In our case, we have a discrete probability distribution over a set of samples $x_i$:
$$F(x_i) = P(X \leq x_i) = \sum_{j=1}^i p(x_j)$$

<figure>
    <img src="https://programmathically.com/wp-content/uploads/2021/03/gamma-pdf-cdf-3-2-971x1024.png" alt="Cumulative Distribution Function">
    <figcaption>
        <p align="center">
            <i>PDF and CDF of a gamma distribution</i>
            <a href="https://programmathically.com/gamma-distribution/">from Programmathically</a>
        </p>
    </figcaption>
</figure>

Notice how the range of the CDF is always between 0 and 1. This is because the CDF is the probability that a random variable will take on a value less than or equal to $x$, and the probability of this happening is always between 0 and 1.

> ##### Intuition
> Also note that when the PDF is higher, the CDF is steeper. If we were then to sample from the range of the CDF (you can visualize this by imagining tracing horizontal lines and seeing where they intersect the CDF), we would be more likely to sample from the steeper parts of the CDF, which correspond to the higher probability regions of the PDF.

#### Search
In order to sample from the probability distribution using the CDF, we generate a uniform random number between 0 and 1 and then find the corresponding value in the CDF. We can do this by searching through the CDF for the first value that is greater than or equal to the random number.

### Final Intuition
One can think of inverse transform sampling as choosing values in the range of the CDF (the range of a CDF is always between 0 and 1) and finding the corresponding value in the domain (the set of samples). Hence, it is called *inverse* transform sampling, as we map from the range to the domain.

### Implementation: `sample_hierarchical`
| Parameter | Type | Description |
| --- | --- | --- |
| `rays` | `Tensor` | A tensor of shape `[num_rays, image_width, image_height, 3]` containing the origins and directions of the rays. |
| `t` | `Tensor` | A tensor of shape `[num_samples_stratified]` containing the $t$ values generated by stratified sampling|
| `deltas` | `Tensor` | A tensor of shape `[num_samples_stratified - 1]` containing the differences between consecutive $t$ values (stratum widths). |
| `weights` | `Tensor` | A tensor of shape `[num_rays, image_width, image_height, num_samples - 1]` containing the weights of each stratum. |
| `num_samples` | `int` | The number of hierarchical samples to generate. |

From our discrete set of $\delta$ values and their corresponding weights, we can generate a probability mass function. We do this by normalizing the weights so that they sum to 1 (as all probabilities in a PMF must sum to 1):
```py
pmf = (weights / torch.sum(weights, dim=-1, keepdim=True))
```

Next, we compute the cumulative distribution function (CDF) of the PMF by taking the cumulative sum of the normalized weights:
```py
cdf = torch.cumsum(pmf, dim=0)
```

Then, we generate a uniform random number between 0 and 1 for each ray:
```py
u = torch.rand(cdf.shape[:-1] + (num_samples,))
```

Finally, we do a search using [`torch.searchsorted`](https://pytorch.org/docs/stable/generated/torch.searchsorted.html) on the CDF to find the corresponding sample indices:
```py
indices = torch.searchsorted(cdf, u)
```

We must clamp the indices to ensure that they are within the bounds of the CDF:
```py
indices = torch.clamp(indices, 0, cdf.shape[-1] - 1)
```

We then use the indices to select the corresponding $t$ and $\delta$ values:
```py
t = torch.gather(t, dim=-1, index=indices)
deltas = torch.gather(deltas, dim=-1, index=indices)
```

Then, we perturb the $t$ values by adding a random number between 0 and the corresponding $\delta$:
```py
t += torch.rand_like(t) * deltas
```

Finally, we compute the position of each sample along the ray by multiplying the direction of the ray by the $t$ value and adding the origin:
```py
return rays_o[..., None, :] + t[..., None] * rays_d[..., None, :], t
```

> We also return the $t$ values of each sample for use in volume rendering.

## Next Steps
After we generate the hierarchical samples, we return to the training loop and pass our points (and directions) into the MLP. The output is then used to volume render the scene.