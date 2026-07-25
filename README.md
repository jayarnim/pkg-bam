# Bayesian Attention Modules Implementation Package

```bash
# INSTALL DEPENDENCIES
conda env create -f env/environment.yaml
conda activate bam
```

```py
# LOAD PKG
import bam
```

- seminar date: 2024.09.19.

## Summary

- attention score is a random variable, not a constant:

$$
\alpha \sim P
$$

- attention score function output means log expected value of attention weight distribution, not attention score:

$$\begin{aligned}
f(q,k)
&=\log{\mathbb{E}\left[\alpha\right]}
\end{aligned}$$

- log expected value of variational distribution is an output of attention score function, taking both q and k as inputs:

$$\begin{aligned}
\log{\mathbb{E}_{Q}\left[\alpha\right]}
&=f(q,k)
\end{aligned}$$

- expected value of prior distibution is an output of fixed function, taking only k as input. it means that prior information is global importance of key:

$$\begin{aligned}
\log{\mathbb{E}_{P}\left[\alpha\right]}
&=g(k)\\
&=\mathrm{softmax}\left[h^{T}\left(W\cdot k + b\right)\right]
\end{aligned}$$

- variational distribution is a distribution that constrains the value of the random variable to be positive, to use linear simplex projection:

$$\begin{aligned}
Q(\alpha)
&:=\mathrm{LogNormal}(\mu,\sigma^{2})
\end{aligned}$$

- the prior distribution is set to a distribution that allows for the calculation of the Kullback-Leibler divergence (KLD) with the variational distribution:

$$\begin{aligned}
P(\alpha)
&:=\mathrm{LogNormal}(\mu,\sigma^{2})
\end{aligned}$$

- when the variational distribution and its prior distribution are set as log-normal distributions, scale parameter $\sigma$ is set as a hyperparameter. thus location parameter $\mu$ is updated as follows:

$$\begin{aligned}
\mu_{Q}
&=f(q,k)-\frac{\sigma^{2}}{2}\\
\mu_{P}
&=g(k)-\frac{\sigma^{2}}{2}
\end{aligned}$$

## Attention Score Function

- `dot`: Luong, M. T., Pham, H., & Manning, C. D. (2015, September). Effective approaches to attention-based neural machine translation. In Proceedings of the 2015 conference on empirical methods in natural language processing (pp. 1412-1421).

$$\begin{aligned}
f(q,k)
&=q^{T}k
\end{aligned}$$

- `scaled`: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

$$\begin{aligned}
f(q,k)
&=\frac{q^{T}k}{\sqrt{d_{k}}}
\end{aligned}$$

- `prod`: He, X., He, Z., Song, J., Liu, Z., Jiang, Y. G., & Chua, T. S. (2018). NAIS: Neural attentive item similarity model for recommendation. IEEE Transactions on Knowledge and Data Engineering, 30(12), 2354-2366.

$$\begin{aligned}
f(q,k)
&=h \cdot \mathrm{ReLU}(W \cdot [p \odot q] + b)
\end{aligned}$$

- `cat`: He, X., He, Z., Song, J., Liu, Z., Jiang, Y. G., & Chua, T. S. (2018). NAIS: Neural attentive item similarity model for recommendation. IEEE Transactions on Knowledge and Data Engineering, 30(12), 2354-2366.

$$\begin{aligned}
f(q,k)
&=h \cdot \mathrm{ReLU}(W \cdot [p \oplus q] + b)
\end{aligned}$$

## Simplex Projection Function

- `linear`: Fan, X., Zhang, S., Chen, B., & Zhou, M. (2020). Bayesian attention modules. Advances in Neural Information Processing Systems, 33, 16362-16376.

$$\begin{aligned}
w
&=\frac{\alpha}{\sum{\alpha}}
\end{aligned}$$

- `smoothing`: He, X., He, Z., Song, J., Liu, Z., Jiang, Y. G., & Chua, T. S. (2018). NAIS: Neural attentive item similarity model for recommendation. IEEE Transactions on Knowledge and Data Engineering, 30(12), 2354-2366.

$$\begin{aligned}
w
&=\frac{\exp{(\alpha)}}{\left[\sum{\exp{(\alpha)}}\right]^{\beta}}
\end{aligned}$$