### 3.2 Adding constraints to existing optimisation objective

Intuition:

Adjust objective function by adding constraints to avoid both disparate treatment and disparate impact via achieve p%-rule.

> A decision making process is said to have **disparate treatment** if its decisions are (partly) based on the subject's sensitive attribute information and it has **disparate impact** if its outcomes disproportionately hurt people with certain sensitive attribute values. (e.g. females, blacks.)

> p%-rule:
>
> ​														$$\min(\frac{P(\text{positive}\mid z=1)}{P(\text{positive}\mid z=0)},\frac{P(\text{positive} \mid z=0)}{P(\text{positive}\mid z=1)}) \ge p\%$$



Notations:

$$\mathbf{x} \in \mathbb{R}^d$$: user feature vectors

$$y \in \{-1,1\}$$: class labels

$$L(\theta)$$: loss function, $$\theta^*$$ denotes optimal solution

$$f(\mathbf{x})$$: mapping function from $$\mathbf{x}$$ to $$y$$

prediction results: $$\begin{cases} d_{\theta^*}(\mathbf{x}_i) \ge 0, \hat{y}_i = 1 \\ d_{\theta^*}(\mathbf{x}_i)<0,\hat{y}_i=-1 \end{cases} $$

$$\{\mathbf{z}_i\}^N_{i=1}$$: sensitive attributes, the fairness constraints will be used for this feature

$$\{\mathbf{x}_i\}^N_{i=1}$$: attributes for classification



Note: feature sets in $$\{\mathbf{z}_i\}^N_{i=1}$$ and $$\{\mathbf{x}_i\}^N_{i=1}$$ are disjoint => comply with disparate treatment criterion

The authors used covariance between $$\{\mathbf{z}_i\}^N_{i=1}$$ and $$\{d_{\theta}(\mathbf{x}_i)\}^N_{i=1}$$ as a measure of decision boundary (un)fairness:

​												$$\begin{align} \text{Cov}(\mathbf{z},d_\theta(\mathbf{x})) &= \mathbb{E}[(\mathbf{z}-\bar{\mathbf{z}})d_\theta(\mathbf{x})]-\mathbb{E}[(\mathbf{z}-\bar{\mathbf{{z}}})]\bar{d}_\theta(\mathbf{x})\\ &\approx \frac{1}{N}\sum^N_{i=1}(\mathbf{z}_i-\bar{\mathbf{z}})d_\theta(\mathbf{x}_i) \end{align}$$

If the decision boundary satisfies 100%-rule:

​												$$P(d_\theta(\mathbf{x})\ge0\mid z=0) = P(d_\theta(\mathbf{x})\ge0\mid z=1)$$

then the empirical covariance will be approximately equal to 0 for a sufficiently large training set.



Then we can write down the objective function:

**Minimising loss function under fairness constraints**

​												 min		$$L(\theta)$$

​												 s.t. 		 $$\text{Cov}(\mathbf{z},d_\theta(\mathbf{x})) \le \mathbf{c}$$

​																$$\text{Cov}(\mathbf{z},d_\theta(\mathbf{x})) \ge -\mathbf{c}$$

Now, the objective becomes Pareto optimal problem: Multi-objective optimisation

we want to:

- minimise the loss function to achieve higher accuracy (performance)
- minimise $$\mathbf{c}$$ to achieve fairness

So now, $$\mathbf{c}$$ becomes the trade-off between fairness and accuracy. If we decrease $$\mathbf{c}$$ towards to 0, then we can get larger p% value, but the accuracy will suffer from this.



The problem with the objective function above is:

If the correlation between sensitive attributes and class labels is very high, then enforcing fairness constraints would make undesirable prediction performance. To accomodate with such situation, the author proposed an alternative formulation:

**Maximising fairness under accuracy constraints**

​												 min		$$\mid \text{Cov}(\mathbf{z},d_\theta(\mathbf{x})) \mid$$

​												 s.t. 		 $$L(\theta)\le(1+\gamma)L(\theta^*)$$

​																 $$\gamma \ge0$$

$$L(\theta^*)$$ is the optimal loss under unconstrained classifier. $$1+\gamma$$ denotes the additional loss if we maximise fairness. If we set $$\gamma =0$$, we can ensure maximise fairness without any loss.



\* The methods can be applied to any convex margin-based classifiers. The authors used SVM and Logistic Regression in the original paper.



Original paper: [Zafar et al. AISTATS 2017](https://arxiv.org/pdf/1507.05259.pdf)

[DEMO](https://github.com/coxxxxx/fair-classification/tree/master/disparate_impact)
