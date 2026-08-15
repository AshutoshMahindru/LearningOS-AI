# M05 dataset policy

M05 generates its numeric workload inside the notebook with `numpy.random.default_rng(20260815)`. No external dataset or runtime download is required.

The three-order teaching fixture is written inline so manual calculations remain inspectable. The benchmark fixture uses 200,000 orders and four products: large enough to reveal Python-loop overhead while remaining bounded for CPU-only laptop execution. Data generation happens outside both timed functions so the comparison measures the two calculation strategies rather than unequal setup work.
