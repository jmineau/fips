Moving from fips 0.1
====================

fips 0.2 replaces the pandas-backed structures of 0.1 (``Block``, ``Vector``,
``Matrix``, ``CovarianceMatrix``, ``InverseProblem`` and the estimators) with
labeled linear algebra: a :class:`~fips.Space` of labels, values on it
(:class:`~fips.Vector`), matrices between spaces (:class:`~fips.Operator`,
:class:`~fips.Covariance`), a :class:`~fips.Problem` that checks its labels
once, and :func:`fips.solve`, which returns a :class:`~fips.Posterior`. Each
table below gives a name of 0.1.0b9 and what replaces it, by topic. The numbers
do not change: the analytic method gives the posterior the Bayesian estimator
gave.

Five changes of behavior to know before the tables:

- **The inputs are matched by label, exactly.** A forward operator, a
  covariance or an offset must have exactly the labels of the observations or
  of the prior, in any order. ``InverseProblem`` filled the labels a matrix
  lacked with zeros (with a "Partial overlap" warning) and dropped the extra
  ones; a :class:`~fips.Problem` raises and names them. A forward operator that
  covers only some cells is put on the whole state by :meth:`Operator.from_blocks
  <fips.Operator.from_blocks>` (a label it lacks is zero), or the state is cut
  to what it covers with :meth:`Problem.restrict <fips.Problem.restrict>`.
- **Labels are kept as given.** 0.1 converted labels to numbers (site codes
  ``"010"`` became ``10``) and ``InverseProblem`` rounded float labels to 6
  decimals. Neither happens now; round on purpose with ``round_labels(6)``.
- **Covariances take standard deviations.** The error components took
  variances; a model's ``with_std`` takes their square roots.
- **The problem's arguments are in a new order and have new names**:
  ``Problem(prior, prior_cov, obs, obs_cov, forward, offset=None)``, where
  ``InverseProblem`` was ``(obs, prior, forward_operator, modeldata_mismatch,
  prior_error, constant=None)``. Pass them by name.
- **A problem is not solved in place.** ``fips.solve(problem)`` returns the
  posterior and leaves the problem as it was.

Pickles written by 0.1 do not load in 0.2. The words of the new names are
defined in :doc:`terminology`.

Vectors
-------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1
     - 0.2
   * - ``Block(series, name="flux")``
     - ``Vector.from_series(series, block="flux")``
   * - ``Vector([block_a, block_b])``
     - ``Vector.from_blocks({"a": series_a, "b": series_b})``
   * - ``Vector(series)`` (a ``block`` level)
     - ``Vector.from_series(series)``
   * - ``Vector(array, index=index)``
     - ``Vector(array, space)``, such as ``Vector(array, Space.from_index(index))``
   * - ``v["flux"]`` (a Series), ``v.blocks["flux"]`` (a ``Block``)
     - ``v["flux"]`` (a :class:`~fips.Vector`; ``.to_series()`` for pandas)
   * - ``v.data``, ``v.to_series()``
     - ``v.to_series()`` (with the ``block`` level first)
   * - ``v.values``, ``v.to_numpy()``
     - ``v.values`` (read-only)
   * - ``v.shape``
     - ``len(v)``
   * - ``v.reindex(index)``
     - ``v.align(space)`` (the same labels, in the order of ``space``)
   * - ``v.round_index(decimals)``
     - ``v.round_labels(decimals)``
   * - ``v.xs(...)``
     - ``v[block]``, or ``v.to_series().xs(...)``
   * - ``v.copy()``
     - not needed: a vector is read-only
   * - ``v.index``, ``v.name``, ``v.to_xarray()``
     - the same

Matrices
--------

``m`` a 0.1 matrix, ``op`` an :class:`~fips.Operator`.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1
     - 0.2
   * - ``Matrix(frame)``, ``ForwardOperator(frame)``
     - ``Operator.from_frame(frame)``
   * - ``MatrixBlock(frame, row_block="co2", col_block="flux")``
     - ``Operator.from_frame(frame, row_block="co2", col_block="flux")``
   * - ``Matrix([MatrixBlock(...), ...])``
     - ``Operator.from_blocks({("co2", "flux"): frame, ...})``
   * - ``Matrix(array, index=index, columns=columns)``
     - ``Operator(array, rows, cols)`` (``rows`` and ``cols`` are :class:`~fips.Space` objects, such as ``Space.from_index(index)``)
   * - ``CovarianceMatrix(frame)``
     - ``Covariance.from_frame(frame)``
   * - ``CovarianceMatrix(series)`` (a diagonal of variances)
     - ``Covariance.diagonal(variances, space)``, or ``Covariance.diagonal(vector)``
   * - ``m.data``, ``m.to_frame()``
     - ``op.to_frame()`` (dense)
   * - ``m.values``, ``m.to_numpy()``
     - ``op.dense()``, or ``np.asarray(op)``
   * - ``m.index``, ``m.columns``
     - ``op.rows.index``, ``op.cols.index`` (``op.rows`` and ``op.cols`` are the spaces)
   * - ``ForwardOperator.obs_index``, ``.state_index``
     - ``op.rows.index``, ``op.cols.index``
   * - ``m["co2", "flux"]`` (a DataFrame), ``m.blocks["co2", "flux"]``
     - ``op["co2", "flux"]`` (an :class:`~fips.Operator`; ``.to_frame()`` for pandas)
   * - ``m.reindex(index, columns)``
     - ``op.align(rows, cols)``
   * - ``m.round_index(decimals)``
     - ``op.round_labels(decimals)``
   * - ``m.scale(c)``
     - ``op * c``
   * - ``m.xs(...)``
     - ``op[row_block, col_block]``, or ``op.to_frame().xs(...)``
   * - ``m.is_sparse``
     - ``isinstance(op.data, fips.linalg.SparseOperator)``
   * - ``m.to_sparse()``, ``m.to_dense()``
     - ``Operator(scipy.sparse.csr_array(op.dense()), op.rows, op.cols)``, ``Operator(op.dense(), op.rows, op.cols)``
   * - ``m.to_xarray()``
     - none: ``op.to_frame()``
   * - ``S.variances``
     - ``S.variances`` (a :class:`~fips.Vector`)
   * - ``S.force_symmetry()``
     - ``(S + S.T) * 0.5`` (the mean of the two triangles)
   * - ``convolve(state, H)``, ``H.convolve(state)``
     - ``H @ vector`` (a :class:`~fips.Vector`)

Covariances
-----------

``v`` variances, ``s = np.sqrt(v)`` the standard deviations, ``space`` the labels.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1
     - 0.2
   * - ``DiagonalError(name, v).build(index)``
     - ``Independent().with_std(s).build(space)``, or ``Covariance.diagonal(v, space)``
   * - ``BlockDecayError(name, v, groupers, ConstantCorrelation())``
     - ``Constant(within=groupers).with_std(s)``
   * - ``BlockDecayError(name, v, groupers, RaggedTimeDecay("obs_time", "2.8h"))``
     - ``Exponential("obs_time", "2.8h", within=groupers).with_std(s)``
   * - a grouper ``index.get_level_values("obs_time").date``
     - ``"obs_time.date"`` in ``within``
   * - ``KroneckerError(name, v, [("time", GridTimeDecay("30D")), (["lat", "lon"], GridSpatialDecay("lat", "lon", 20.0))])``
     - ``Kronecker(Exponential("time", "30D"), Exponential(["lat", "lon"], 20.0, metric="haversine")).with_std(s)`` (in any order of the levels)
   * - ``CovarianceBuilder([a, b, c]).build(index)``
     - ``(a + b + c).build(space)``
   * - ``component.build(index)`` (a DataFrame)
     - ``model.build(space).to_frame()``; or give the model to the :class:`~fips.Problem`, which builds it
   * - ``BlockDecayError.build(index, n_jobs=...)``
     - ``build(space)``: sequential, and sparse between groups
   * - a Series of variances on some of the levels
     - an array of standard deviations in the space's order (join the Series to the labels yourself), or a :class:`~fips.Vector`
   * - ``ConstantCorrelation()``
     - ``Constant()``
   * - ``RaggedTimeDecay(level, scale)``, ``GridTimeDecay(scale)``
     - ``Exponential(level, scale)``
   * - ``GridSpatialDecay("lat", "lon", scale)``
     - ``Exponential(["lat", "lon"], scale, metric="haversine")``
   * - a ``decay_func``
     - ``Gaussian``, ``Matern``, or a subclass of :class:`~fips.covariance.CorrelationModel`
   * - a subclass of ``ErrorComponent``
     - a subclass of :class:`~fips.covariance.CorrelationModel`
   * - ``fips.metrics.haversine_matrix``, ``time_diff_matrix``
     - ``fips.covariance.distances.haversine_matrix``, ``time_diff_matrix``

Solving
-------

``problem`` a :class:`~fips.Problem`, ``post = fips.solve(problem)``.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1
     - 0.2
   * - ``InverseProblem(obs, prior, forward_operator, modeldata_mismatch, prior_error, constant)``
     - ``Problem(prior=..., prior_cov=..., obs=..., obs_cov=..., forward=..., offset=...)``
   * - a Series or DataFrame as an input
     - a :class:`~fips.Vector` (``Vector.from_series``) or an :class:`~fips.Operator` (``Operator.from_frame``)
   * - ``prior_error=`` a Series of variances
     - ``prior_cov=Covariance.diagonal(variances, prior.space)``, or a covariance model
   * - ``constant=``
     - ``offset=`` (a :class:`~fips.Vector` on the observations, or a number)
   * - ``round_index=6``
     - ``round_labels(6)`` on each input, on purpose
   * - labels a matrix lacks, filled with zeros
     - ``Operator.from_blocks({("co2", "flux"): J}, rows=obs.space, cols=prior.space)``
   * - ``problem.solve("bayesian", gamma=g)``
     - ``post = fips.solve(problem, gamma=g)`` (``"analytic"`` by default)
   * - ``available_estimators()``
     - ``fips.available_methods()``
   * - an ``Estimator`` subclass, ``ESTIMATOR_REGISTRY.register``
     - a method module (see "Writing a method" in :doc:`reference/posterior`)
   * - ``problem.forward_operator``, ``.prior_error``, ``.modeldata_mismatch``, ``.constant``
     - ``problem.forward``, ``.prior_cov``, ``.obs_cov``, ``.offset``
   * - ``problem.state_index``, ``.obs_index``
     - ``problem.prior.index``, ``problem.obs.index``
   * - ``problem.n_state``, ``.n_obs``
     - ``len(problem.prior)``, ``len(problem.obs)``
   * - ``problem.get_block("prior", "flux")``
     - ``problem.prior["flux"]``, ``problem.prior_cov["flux", "flux"]``
   * - ``problem.estimator``
     - ``post`` (and ``post.options``)
   * - ``problem.posterior``, ``est.x_hat``
     - ``post.estimate``
   * - ``problem.posterior_error``, ``est.S_hat``
     - ``post.cov()`` (``post.variances`` for its diagonal)
   * - ``problem.posterior_obs``, ``est.y_hat``
     - ``post.modeled_obs``
   * - ``problem.prior_obs``, ``est.y_0``
     - ``problem.predict(problem.prior)``
   * - ``problem.prior_obs_error``, ``est.S_y0`` (:math:`H S_0 H^T`)
     - ``problem.forward @ problem.prior_cov @ problem.forward.T``; ``post.innovation_cov`` adds :math:`S_z / \gamma`
   * - ``problem.kalman_gain``, ``est.K``
     - ``post.kalman_gain``
   * - ``problem.averaging_kernel``, ``est.A``
     - ``post.averaging_kernel()``
   * - ``problem.desroziers``, ``est.desroziers`` (an :math:`m \times m` matrix)
     - ``post.desroziers(by=None)`` (a DataFrame: diagnosed, specified, ratio)
   * - ``est.z``, ``est.x_0``, ``est.c``
     - ``problem.obs.values``, ``problem.prior.values``, ``problem.offset``
   * - ``est.H``, ``est.S_0``, ``est.S_z``
     - ``problem.forward.dense()``, ``problem.prior_cov.dense()``, ``problem.obs_cov.dense()`` (``.data`` for the stored form)
   * - ``est.DOFS``, ``est.R2``, ``est.RMSE``
     - ``post.dofs``, ``post.r2``, ``post.rmse``
   * - ``est.reduced_chi2``, ``est.uncertainty_reduction``
     - ``post.reduced_chi2``, ``post.uncertainty_reduction``
   * - ``est.U_red``
     - ``100 * (1 - np.sqrt(post.variances.values / problem.prior_cov.variances.values))``
   * - ``est.leverage(x)``
     - ``post.leverage(vector)`` (an :class:`~fips.Operator` on the observations)
   * - ``est.forward(x)``
     - ``problem.predict(vector)`` (:math:`H x + c`)
   * - ``est.residual(x)``
     - ``problem.obs.values - problem.predict(vector).values``
   * - ``est.cost(x)``
     - ``obs_cov.quadratic_form`` and ``prior_cov.quadratic_form`` of the two misfits; ``post.chi2`` at the estimate

Aggregation
-----------

``W`` an aggregation from :mod:`fips.aggregate`.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1
     - 0.2
   * - ``ObsAggregator(level="obs_time", freq="D")``
     - ``fips.aggregate.mean(obs.space, by=[<the other levels>, pd.Grouper(level="obs_time", freq="D")])``
   * - ``ObsAggregator(by="site", func="sum")``
     - ``fips.aggregate.sum(obs.space, by="site")``
   * - ``ObsAggregator(by=callable)``
     - a Series that maps a level's values to groups, or a ``pd.Grouper``
   * - ``ObsAggregator(..., blocks="co2")``
     - ``fips.aggregate.mean(..., blocks="co2")`` (other blocks pass through)
   * - ``agg.apply(obs, forward_operator, modeldata_mismatch, constant)``
     - ``problem.aggregate(W)`` (a new :class:`~fips.Problem`)
   * - ``integrate_over_time_bins(series, bins, time_dim="time")``
     - ``fips.aggregate.sum(v.space, by=[bins.rename("time"), <the other levels>]) @ v``
   * - ``pipeline.filter_state_space(obs, prior)``
     - choose what to keep in your code, then ``problem.restrict(keep)``
   * - ``fips.filters.enough_obs_per_interval``, ``select_intervals_with_min_obs``
     - a mask over ``obs.to_series()`` in your code

The flux subpackage and the pipelines
-------------------------------------

fips knows no science domain: a flux inversion is a :class:`~fips.Problem` whose blocks are named ``flux`` and ``concentration``, read by name. Each accessor below gives a fips object; ``.to_series()`` or ``.to_frame()`` gives pandas.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1
     - 0.2
   * - ``FluxProblem(obs, prior, forward_operator, modeldata_mismatch, prior_error, constant)``
     - ``Problem(prior=..., prior_cov=..., obs=..., obs_cov=..., forward=..., offset=...)``
   * - ``FluxProblem.solve()``
     - ``post = fips.solve(problem)``
   * - ``problem.concentrations``
     - ``problem.obs["concentration"]``
   * - ``problem.enhancement``
     - ``problem.obs["concentration"].values - problem.offset["concentration"].values``
   * - ``problem.background``
     - ``problem.offset["concentration"]``
   * - ``problem.prior_fluxes``
     - ``problem.prior["flux"]``
   * - ``problem.jacobian``
     - ``problem.forward["concentration", "flux"]``
   * - ``problem.prior_flux_error``
     - ``problem.prior_cov["flux", "flux"]``
   * - ``problem.concentration_error``
     - ``problem.obs_cov["concentration", "concentration"]``
   * - ``problem.prior_concentrations``
     - ``problem.predict(problem.prior)["concentration"]``
   * - ``problem.prior_concentration_error``
     - ``(problem.forward @ problem.prior_cov @ problem.forward.T)["concentration", "concentration"]``
   * - ``problem.posterior_fluxes``
     - ``post.estimate["flux"]``
   * - ``problem.posterior_flux_error``
     - ``post.cov()["flux", "flux"]``
   * - ``problem.posterior_concentrations``
     - ``post.modeled_obs["concentration"]``
   * - ``InversionPipeline``, ``FluxInversionPipeline`` (``get_obs``, ``get_prior``, ``get_forward_operator``, ``get_prior_error``, ``get_modeldata_mismatch``, ``get_constant``, ``get_inputs``)
     - functions in your code that build the :class:`~fips.Problem`
   * - ``pipeline.aggregate_obs_space(...)``
     - ``problem.aggregate(W)``
   * - ``pipeline.run(estimator_kwargs=...)``
     - ``post = fips.solve(problem, **options)``
   * - ``FluxInversionPipeline.summarize()`` (printed)
     - ``post.summary()`` (a one-row DataFrame, returned)

Plotting
--------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1
     - 0.2
   * - ``problem.plot.fluxes()`` (``FluxPlotter``)
     - ``post.estimate["flux"].to_xarray().mean("time").plot(x="lon", y="lat")``
   * - ``problem.plot.concentrations()``
     - ``plot_comparison`` of one site's ``problem.obs``, ``problem.predict(problem.prior)`` and ``post.modeled_obs`` (see :doc:`examples/flux_regional`)
   * - ``plot_comparison(block, ...)``
     - ``plot_comparison(vector, ...)``, or Series; errors as a :class:`~fips.Covariance` or an :class:`~fips.Operator`
   * - ``compute_credible_interval(samples, q=(0.05, 0.95))``
     - ``np.quantile(samples, q, axis=0)``
   * - ``plot_error_norm(prior, posterior, truth)``
     - ``np.linalg.norm(prior - truth, axis=1)`` and the same for the posterior, plotted with matplotlib

Saving
------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1
     - 0.2
   * - ``problem.to_file(path)``, ``matrix.to_file(path)``
     - ``pickle.dump(problem, file)``
   * - ``InverseProblem.from_file(path)``
     - ``problem = pickle.load(file)``
   * - a 0.1 pickle
     - does not load: rebuild the problem

Import paths and extras
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1
     - 0.2
   * - ``fips.Block``, ``fips.Matrix``, ``fips.MatrixBlock``, ``fips.ForwardOperator``, ``fips.CovarianceMatrix``
     - ``fips.Vector``, ``fips.Operator``, ``fips.Covariance``
   * - ``fips.InverseProblem``, ``fips.Estimator``, ``fips.InversionPipeline``, ``fips.convolve``
     - ``fips.Problem``, ``fips.solve``; see the tables above
   * - ``fips.problem.InverseProblem``, ``fips.estimators``
     - ``fips.problem.Problem``, ``fips.methods`` (``fips.solve``), ``fips.posterior.Posterior``
   * - ``fips.vector.Block``, ``fips.matrix``, ``fips.operators``
     - ``fips.vector.Vector``, ``fips.space.Space``, ``fips.operator.Operator``
   * - ``fips.covariance`` (the error components), ``fips.kernels``
     - ``fips.covariance`` (``Covariance`` and the models; the kernels are also ``fips.Exponential``, ...)
   * - ``fips.metrics``
     - ``fips.covariance.distances``
   * - ``fips.aggregators``
     - ``fips.aggregate``
   * - ``fips.problems.flux``, ``fips.pipeline``, ``fips.filters``, ``fips.base``, ``fips.indexes``
     - none (see the tables above)
   * - ``pip install "fips[flux]"``
     - ``pip install "fips[viz]"`` (matplotlib only)
