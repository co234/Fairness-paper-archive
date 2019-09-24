### Variational Fair Autoencoder

The target value is to remove sensitive attribute $$\mathbf{S}$$ and learn a

$$\mathbf{y},\mathbf{z}_2 \sim \text{Cat}(\mathbf{y})P(\mathbf{z}_2)$$
$$\mathbf{z}_1 \sim P_{\theta}(\mathbf{z}_1 \mid \mathbf{z}_2, \mathbf{y})$$
$$\mathbf{x} \sim P_{\theta}(\mathbf{x} \mid \mathbf{z}_1, \mathbf{s})$$



Posterior:

$$q_{\phi}(\mathbf{z}_1,\mathbf{z}_2,\mathbf{y} \mid \mathbf{x},\mathbf{s})$$



$$\begin{align} q_{\phi}(\mathbf{z}_{1n},\mathbf{z}_{2n},\mathbf{y}_n \mid \mathbf{x}_n,\mathbf{s}_n) &= q_{\phi}(\mathbf{z}_{2n},\mathbf{y}_n \mid \mathbf{z}_{1n},\mathbf{x}_n,\mathbf{s}_n)q_{\phi}(\mathbf{z}_{1n}\mid \mathbf{x}_n, \mathbf{s}_n)\\ &= q_{\phi}(\mathbf{z}_{2n} \mid \mathbf{z}_{1n},\mathbf{y}_n)q_{\phi}(\mathbf{y}_n \mid \mathbf{z}_{1n})q_{\phi}(\mathbf{z}_{1n}\mid \mathbf{x}_{n},\mathbf{s}_n) \end{align}$$





where
