# Non-Experts

If this project survived, I will post codes for the "non-experts" problem.


## Bug History

- Use PCA instead of linear algebra.
- Forget to normalize $\theta$ at the beginning. However, normalized $\widehat{\theta}$ and $\widetilde\theta$ is estimated later.
- $\alpha^{(r)}$ should be different for each annotator. However, I forgot it and let the intercept be the same because the logistic regression by default has only one intercept.
  - However, at present I still failed to handle it beautifully. It seems that I should set some constraints on the intercept $\alpha^{(r)}$. My requirements are not support by standard logistic codes offered by sklearn. I should write my own codes.