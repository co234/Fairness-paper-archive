Variational Fair Autoencoder



The intuitive idea is to learn fair representations by extracting the most informative hidden factors $$\mathbf{z}$$ and removing uninformative factors (sensitive information or nuisance variables $$\mathbf{s}$$). So the target is to separate $$\mathbf{z}$$ and $$\mathbf{s}$$ from the data. We can do this by using a factorised priors $$p(\mathbf{s})p(\mathbf{z})$$ (this means independent). However, some dependencies may still remain (via variational posterior $$q(\mathbf{z}\mid\mathbf{x},\mathbf{s})$$). Thus the authors use "Maximum Mean Discrepancy" to penalise differences between all order moments of the marginal posterior distribution $$q(\mathbf{z}\mid \mathbf{s}=k)$$ and $$q(\mathbf{z}\mid\mathbf{s}=k')$$.



VAE architecture:

**Encoder**

Input: $$\mathbf{x}$$ and $$\mathbf{s}$$

Variational posterior of encoder is $$q_\phi(\mathbf{z}\mid\mathbf{x},\mathbf{s})$$, parameters $$\phi$$



**Decoder**

Input: $$\mathbf{z}$$ and $$\mathbf{s}$$

Conditional probability of decoder is $$p_\theta(\mathbf{x}\mid \mathbf{z},\mathbf{s})$$, parameters $$\theta$$



The data likelihood is given by:

$$p_\theta(\mathbf{x}\mid\mathbf{s}) = \int p_\theta(\mathbf{z})p_\theta(\mathbf{x}\mid\mathbf{z},\mathbf{s})d\mathbf{z}$$

We want to resolve the dependencies between $$\mathbf{x}$$ and $$\mathbf{s}$$, but the likelihood is intractable.



Work with the log data likelihood to equip encoder and decoder.

$$\begin{align}\sum^N_{n=1}\log p(\mathbf{x}_n\mid\mathbf{s}_n) &=\sum^N_{n=1}\mathbb{E}_{\mathbf{z}_n\sim q_\phi(\mathbf{z}_n\mid\mathbf{x_n},\mathbf{s_n})}[\log p_\theta(\mathbf{x}_n\mid \mathbf{s}_n)]\\&= \sum^N_{n=1}\mathbb{E}_{\mathbf{z}_n}[\log \frac{p_\theta(\mathbf{x}_n\mid \mathbf{s_n},\mathbf{z_n})p_\theta(\mathbf{z})}{p_\theta(\mathbf{z}_n\mid \mathbf{x}_n,\mathbf{s}_n)}] \\ &= \sum^N_{n=1}\mathbb{E}_{\mathbf{z}_n}[\log \frac{p_\theta(\mathbf{x}_n\mid \mathbf{s_n},\mathbf{z_n})p_\theta(\mathbf{z})}{p_\theta(\mathbf{z}_n\mid \mathbf{x}_n,\mathbf{s}_n)}\frac{q_\phi(\mathbf{z}_n\mid\mathbf{x_n},\mathbf{s_n})}{q_\phi(\mathbf{z}_n\mid\mathbf{x_n},\mathbf{s_n})}]\\ &=\sum^N_{n=1}\mathbb{E}_{\mathbf{z}_n}[\log p_\theta(\mathbf{x}_n\mid \mathbf{s}_n,\mathbf{z}_n)]- \mathbb{E}_{\mathbf{z}_n}[\log \frac{q_\phi(\mathbf{z}_n\mid\mathbf{x}_n,\mathbf{s}_n)}{p_\theta(\mathbf{z})}]+ \mathbb{E}_{\mathbf{z}_n}[\log \frac{q_\phi(\mathbf{z}_n\mid \mathbf{x_n},\mathbf{s}_n)}{p_\theta(\mathbf{z}_n\mid \mathbf{x_n},\mathbf{s}_n)}] \\ &=\sum^N_{n=1}\mathbb{E}_{\mathbf{z}_n}[\log p_\theta(\mathbf{x}_n\mid \mathbf{s}_n,\mathbf{z}_n)]-KL(q_\phi(\mathbf{z}_n\mid\mathbf{x}_n,\mathbf{s}_n)||p_\theta(\mathbf{z}))+KL(q_\phi(\mathbf{z}_n\mid \mathbf{x_n},\mathbf{s}_n)||p_\theta(\mathbf{z}_n\mid \mathbf{x_n},\mathbf{s}_n))\end{align}$$



Because $$KL(q_\phi(\mathbf{z}_n\mid \mathbf{x_n},\mathbf{s}_n)||p_\theta(\mathbf{z}_n\mid \mathbf{x_n},\mathbf{s}_n))) \ge 0$$

$$\sum^N_{n=1}\log p(\mathbf{x}_n\mid\mathbf{s}_n)  \ge\sum^N_{n=1}\mathbb{E}_{\mathbf{z}_n}[\log p_\theta(\mathbf{x}_n\mid \mathbf{s}_n,\mathbf{z}_n)]-KL(q_\phi(\mathbf{z}_n\mid\mathbf{x}_n,\mathbf{s}_n)||p_\theta(\mathbf{z}))$$



We denote the ELBO as $$\mathcal{F}(\phi,\theta;\mathbf{x}_n,\mathbf{s}_n)$$

We choose the variational posterior to be Gaussian:

$$q_\phi(\mathbf{z}_n\mid\mathbf{x}_n,\mathbf{s}_n) = \mathcal{N}(\mathbf{z}_n\mid\mathbf{\mu}_n = f_\phi(\mathbf{x}_n,\mathbf{s}_n),\mathbf{\sigma}_n=e^{f_\phi(\mathbf{x}_n,\mathbf{s}_n)})$$

and

$$p_\theta(\mathbf{x}_n\mid \mathbf{s}_n,\mathbf{z}_n) = f_\theta(\mathbf{s}_n,\mathbf{z}_n)$$











The target value is to remove sensitive attribute $$\mathbf{S}$$ and learn a

$$\mathbf{y},\mathbf{z}_2 \sim \text{Cat}(\mathbf{y})P(\mathbf{z}_2)$$
$$\mathbf{z}_1 \sim P_{\theta}(\mathbf{z}_1 \mid \mathbf{z}_2, \mathbf{y})$$
$$\mathbf{x} \sim P_{\theta}(\mathbf{x} \mid \mathbf{z}_1, \mathbf{s})$$



Posterior:

$$q_{\phi}(\mathbf{z}_1,\mathbf{z}_2,\mathbf{y} \mid \mathbf{x},\mathbf{s})$$



$$\begin{align} q_{\phi}(\mathbf{z}_{1n},\mathbf{z}_{2n},\mathbf{y}_n \mid \mathbf{x}_n,\mathbf{s}_n) &= q_{\phi}(\mathbf{z}_{2n},\mathbf{y}_n \mid \mathbf{z}_{1n},\mathbf{x}_n,\mathbf{s}_n)q_{\phi}(\mathbf{z}_{1n}\mid \mathbf{x}_n, \mathbf{s}_n)\\ &= q_{\phi}(\mathbf{z}_{2n} \mid \mathbf{z}_{1n},\mathbf{y}_n)q_{\phi}(\mathbf{y}_n \mid \mathbf{z}_{1n})q_{\phi}(\mathbf{z}_{1n}\mid \mathbf{x}_{n},\mathbf{s}_n) \end{align}$$





where
