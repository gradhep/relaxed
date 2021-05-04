# List of differentiable operations


## Definitely useful with known solution

* **Classification (binning)**:
Assigning an event to a bin in a histogram or classifying it as a particular class label is a non-differentiable operation. Multi-class classification is a classic example in machine learning and statistics, and is typically relaxed with a sigmoid or a softmax.
  * This was used in INFERNO and neos
  * Alternatively, one could calculate smooth probability assignments using Kernel Density Estimation or some other kernel based approach

* **Differentiable ranking and sorting**:
Sorting is a fundamental operation. For instance, we typically sort particles by $p_T$.

  * Differentiable Ranks and Sorting using Optimal Transport [https://arxiv.org/abs/1905.11885](https://arxiv.org/abs/1905.11885)
  *  O(nlogn) time and O(n) space complexity [https://arxiv.org/abs/2002.08871](https://arxiv.org/abs/2002.08871) and [great slides](https://raw.githubusercontent.com/mblondel/mblondel.github.io/9e103aad534d3e2d51a357c72b2485309131e719/talks/mblondel-CIRM-2020-03.pdf)


* **Differentiable clustering (partitions)**
We have a set of objects and we would like to cluster or partition them. We can think of this in terms of graph where the nodes are the objects and edges indicate two objects are in the same cluster. We want all objects in the same cluster to be connected and no objects in different clusters to be connected.

  * This can be imposed if the adjacency matrix is restricted to be of the form $u u^T$, where $u$ is a softmax output. This was used in [Set2Graph: Learning Graphs From Sets](https://arxiv.org/abs/2002.08772) for vertexing and is also described in slide 27 of [this talk](https://indico.cern.ch/event/809820/contributions/3632659/attachments/1971659/3280030/GNN_NYU_3_Jan_2020.pdf).
  * note: one might think of using something like this for clustering calorimeter cells to calorimeter clusters.

* **Barlow-Beeston for Monte Carlo Statistical Uncertainty:**
The statistical uncertainty on template histograms from limited statistical uncertainty can be dealth with in a clean way by jointly modelling the statistical fluctuations in the data and the statistical fluctuations in the Monte Carlo samples. This was treated in [Fitting using finite Monte Carlo samples](https://doi.org/10.1016/0010-4655(93)90005-W) (pdf from [at FermiLab](https://lss.fnal.gov/archive/other/man-hep-93-1.pdf)). In a simple one-bin example one would model as $P(n,m|\mu,\lambda) = Pois(n|\mu+\lambda)Pois(m|\tau\lambda)$ where $n$ is count in data in a signal region, $\mu$ is the unknown exepected signal rate, $\lambda$ is the unknown expected background rate (a nuisance parameter), $\tau$ is the ratio of the Monte Carlo luminosity to data luminosity, and $m$ is the count in the Monte Carlo sample. This can easily be extended to multiple bins and multiple background sources per bin, but it introduces a nuisance parameter for each component of each bin. Note in this setup the observed Monte Carlo are treated as data (since it fluctuates and is on the left of the "|"). In HistFactory language, the Monte Carlo observation $m$ would be the `Data` of a new `Channel` and the unknown background $\tau\lambda$ would be modeled with a `ShapeFactor` that would be shared with the `Channel` that has the real observed data $n$. This is typically very heavy and leads to a proliferation of nuisance parameters, which cause problems for Minuit. Thus, typically an approximate approach is used where the different background contributions are combined. In HistFactory this is what is done when using `StatErrorConfig`. This treatment is usually fine, but has corner cases when $m=0$. One interesting aspect of the Barlow-Beeston approach is that optimization on the nuisance parameter $\lambda$ decouples from optimization on $\mu$. In fact, there is a closed form solution for $\hat{\lambda}(n,m,\mu)$ (eq. 14), so optimizing the full likelihood can be thought of as a nested optimization with $\lambda$ in the inner loop. Moreover, it can be thought of as the implicit minimization used for the profile likelihood fit in neos. Several years ago George Lewis wrote a wrapper for the log-likeihood created in HistFactory so that $\lambda$ was solved exactly and only the profiled likelihood with $\mu$ was exposed to Minuit. While elegant conceptually, the implementation in RooFit did not lead to significant performance gains for the number of nuisance parameters in the models at that time. However, it would be interesting to revisit this in the context of pyhf and grad-hep. References:

  * [RooBarlowBeestonLL.cxx](https://root.cern/doc/master/RooBarlowBeestonLL_8cxx_source.html) [RooBarlowBeestonLL.h](https://root.cern/doc/master/RooBarlowBeestonLL_8h_source.html)
  * [A RooFit example](https://root.cern/doc/master/rf709__BarlowBeeston_8C.html)

* **ROC AUC:**
While the area under ROC curve (ROC AUC) is not usually our ultimate physics goal, it may be useful or motivated in some cases. The ROC curve is non-differentiable, but can be relaxed into a rank statistic. This was used for example in [Backdrop: Stochastic Backpropagation](https://arxiv.org/abs/1806.01337)
 * Herschtal, A. and Raskutti, B. (2004). Optimising area under the roc curve using gradient descent. In Proceedings of the Twenty-first International Conference on Machine Learning, ICML ’04, pages 49–, New York, NY, USA. ACM. [doi/10.1145/1015330.1015366](https://dl.acm.org/doi/10.1145/1015330.1015366)

## Definitely useful seeking solution

* **Differentiable legend placement in plots:**
They are so annoying aren't they?

* **Differentiable peer review:**
accept/reject is so non-diffable

## Potentially useful

* **Differentiable Feature Selection by Discrete Relaxation**
 See [paper](https://www.microsoft.com/en-us/research/publication/differentiable-feature-selection-by-discrete-relaxation/)

* **Gumbel Max Trick & Gumbel Machinery:**
The Gumbel-Max Trick is a method to sample from a categorical distribution $Cat(\alpha_1, \dots, \alpha_K)$, where category $k$ has $\alpha_k$
probability to be sampled among $K$ categories, and relies on the Gumbel distribution defined by the Cumulative Distribution Function.

  * [Gumbel Max Trick](https://laurent-dinh.github.io/2016/11/22/gumbel-max.html)
  * [Gumbel Machinery](https://cmaddis.github.io/gumbel-machinery)

 * **Sparse Structured Prediction:**
  See paper [Differentiable Relaxed Optimization for Sparse Structured Prediction](https://arxiv.org/abs/2001.04437)

*  **Coreference resolution**:
"Coreference resolution is the task of identifying all mentions which refer to the same entity in a document." "Coreference resolution can be regarded as a clustering problem: each cluster corresponds to a single entity and consists of all its mentions in a given text." From Optimizing Differentiable Relaxations of Coreference Evaluation Metrics [https://arxiv.org/abs/1704.04451](https://arxiv.org/abs/1704.04451)
