# Several Papers on Diffusion Models for Image Synthesis

In this post, I would like to share several papers on diffusion models for image synthesis.

Here is a list of papers I want to share

## DDPM (Denoising Diffusion Probabilistic Models)
Diffusion Models are latent variable models of the form $p_{\theta}(x) = \int p_{\theta}(x_{0:T})dx_{1:T}$, where $x_1,...,x_T$ are latents of the same dimensionality as the data $x_0$. The joint distribution $p_{\theta}(x_{0:T})$ is  called reverse process, and it is defined as a Markov chain with learnd Gaussian transitions starting at $p(x_T) = N(0, I)$.

$$p_{\theta}(x_{0:T}) := p(x_T) \Pi_{t=1}^T p_{\theta}(x_{t-1}|x_t)$$
$$p_{\theta}(x_{t-1}|x_t) \sim N(x_{t-1}; \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t, t)) \tag{1}$$

<!-- ![Diffusion forward and reverse process](/Users/tonyzhang1231/Desktop/pycharm/github_post/tonyz.github.io/figures/DDPM-algo.png) -->

### Forward process
Suppose $x_0$ is a data point sample from the real data distribution $q(x)$. The forward diffusion process is defined as that we add small amount of noise to the $x_0$ in $T$ steps, resulting in $x_1, x_2, ... , x_T$. The variance of noise is controlled by an increasing sequence $\beta_t \in (0,1), t = 1,...,T$. So we have

$$q(x_t|x_{t-1}) = N (x_{t}; \sqrt{1-\beta_t}x_{t-1}, \beta_tI), \hspace{0.5cm} q(x_{1:T}|x_0) = \Pi_{t=1}^T q(x_t|x_{t-1})$$

Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \Pi_{i=1}^t {\alpha}_t$, we can obtain the relationship between $x_0$ and $x_t$ by some math derivation.

$$x_t = \sqrt{\alpha_t}x_{t} + \sqrt{1-\alpha_t}\epsilon_{t-1} = ... = \sqrt{\bar{\alpha}_t}x_{0} + \sqrt{1-\bar{\alpha}_t}\epsilon$$

in other words,

$$q(x_t|x_0) = N (x_{t}; \sqrt{\bar{\alpha}_t}x_{0}, (1-\bar{\alpha}_t)I)$$

Since $\beta_1 < \beta_2 <...< \beta_T$, then $\bar{\alpha}_1 > \bar{\alpha}_2 > ... > \bar{\alpha}_T$.

### Reverse process
The reverse process is also called the denoising process. If we can know the distribution $q(x_{t-1}|x_{t}), t = T,..., 1$, then we are enable start from $x_T \sim N(0, I)$ and sample $x_{T-1}, x_{T-2}$ until $x_0$. Unfortunately this distribtution is intractable. However, it is tractable if conditioning on $x_0$ as well. By using the Bayes' rule, finally we can get

$$q(x_{t-1}|x_{t}, x_0) = N (x_{t-1};  \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_tI)$$

where

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}}\epsilon_t) \tag{2}$$
$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$$
are posterior mean and variance.

So the $p_{\theta}(x_{t-1}|x_t) \sim N(x_{t-1}; \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t, t))$ is used to to approximate $q(x_{t-1}|x_{t})$, that is, we can train a model $\mu_{\theta}(x_t, t)$ to estimate $\tilde{\mu}_t$. From Eq (2), we know $x_t$ is given at time $t$, so we just need to train a model $\epsilon_\theta(x_t, t)$ to estimate $\epsilon_t$.

### Objective function
As mentioned before, we want to train a model $\epsilon_\theta(x_t, t)$, so how to design the objective function? Since we goal is to maximize the log-likelihood of $x_0$ given $x_0 \sim q(x_0)$, so the objective function can be the negative log-likelihood or the cross entropy

$$ L = E_{x_0 \sim q(x_0)} [-\log p_\theta(x_0) ] \tag{3}$$

Simiar to the method used for VAE, we optimize its variational lower bound, since directly optimizing (3) is not tractable.

$$ -\log p_\theta(x_0) <= -\log p_\theta(x_0) + D_{KL} (q (x_{1:T}|x_0)|| p_\theta (x_{1:T}|x_0) ) = E_{q (x_{1:T}|x_0)} [\log\frac{q (x_{1:T}|x_0)}{p_\theta(x_{0:T})}] \tag{4}$$

take the $E_{q(x_0)}$ on both side, we can have
$$L_{VLB} = E_{q (x_{0:T}|x_0)}[\log\frac{q (x_{1:T}|x_0)}{p_\theta(x_{0:T})}] >= E_{q (x_0)}[-\log p_\theta(x_0)] \tag{5} $$

So we aim to minimize $L_{VLB}$, which can be further breakdown as
$$L_{VLB} = L_T + L_{T-1} + ... + L_0$$
where
$$
L_T = D_{KL}(q(x_T|x_0)||p_\theta(x_T)) \\
L_t = D_{KL}(q(x_t|, x_{t+1}, x_0)||p_\theta(x_t|x_{t+1})); 1\le t \le T-1 \\
L_0 = -\log p_\theta(x_0|x_1)
$$
since $p_\theta$ and $q(x_t|x_{t+1}, x_0)$ are both gaussian distributions, their KL have closed form, that is
$$L_t = E_{x_0, \epsilon}[\frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar{\alpha}_t)||\Sigma_\theta||_2^2}||\epsilon_t - \epsilon_\theta(x_t,t)||^2] \tag{6}$$

The loss (6) was simplied in DDPM paper to
$$L_t^{simple} = E_{x_0, \epsilon_t}||\epsilon_t - \epsilon_\theta(x_t,t)||^2] \tag{7}$$

### Training and sampling Algorithm
![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM-algo.png)


## Improved DDPM

## DDIM

## Diffusion Model beats GAN

## Classifier-Free Diffusion Guidance

## Glide

## Imagen

## Dalle2 or UnClip

## Latent Diffusion Model (LDM)

## Some Benchmarks
