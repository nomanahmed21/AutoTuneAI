import random
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm


class BayesianAutoTuneAgent:
    def __init__(self, search_space, train_fn=None, warmup_samples=400, max_workers=None):
        """
        Args:
            search_space: Dict of param_name -> (min, max) | [choices] | fixed_value
            train_fn: Function that takes config dict and returns a score
            warmup_samples: Total number of random samples for warmup
            max_workers: Thread pool size (None = cpu_count * 5)
        """
        self.search_space = search_space
        self.param_keys = sorted(search_space.keys())  # Cache sorted keys
        self.results = []
        self._lock = Lock()  # Thread safety for results

        # GP Model with optimized kernel
        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,  # Better kernel optimization
            random_state=42
        )

        # Parallel warmup
        if train_fn is not None:
            self._parallel_warmup(train_fn, warmup_samples, max_workers)

    def _parallel_warmup(self, train_fn, warmup_samples, max_workers):
        """Run warmup evaluations in parallel."""
        print(f"🚀 Running parallel warmup with {warmup_samples} samples...")

        def evaluate_config(_):
            config = self.random_sample()
            score = train_fn(config)
            return {"config": config, "metrics": score}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_config, i) for i in range(warmup_samples)]

            for future in as_completed(futures):
                result = future.result()
                with self._lock:
                    self.results.append(result)

        print(f"✅ Warmup completed with {len(self.results)} samples")

    def encode(self, config):
        """Encode a single config to numeric array."""
        return [config[k] for k in self.param_keys]

    def encode_batch(self, configs):
        """Vectorized encoding for multiple configs."""
        return np.array([[c[k] for k in self.param_keys] for c in configs])

    def random_sample(self):
        """Generate a random config from the search space."""
        config = {}
        for k, v in self.search_space.items():
            if isinstance(v, tuple):
                config[k] = random.uniform(v[0], v[1])
            elif isinstance(v, list):
                config[k] = random.choice(v)
            else:
                config[k] = v
        return config

    def expected_improvement(self, X, xi=0.001):
        """
        Calculate Expected Improvement with exploration parameter.

        Args:
            X: Candidate points (n_samples, n_features)
            xi: Exploration-exploitation tradeoff (higher = more exploration)
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        best = max(r["metrics"] for r in self.results)

        with np.errstate(divide='ignore', invalid='ignore'):
            imp = mu - best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma < 1e-9] = 0.0  # No improvement where uncertainty is zero

        return ei, mu, sigma

    def bayesian_sample(self, n_candidates=500):
        """Select next config using Expected Improvement."""
        candidates = [self.random_sample() for _ in range(n_candidates)]
        X = self.encode_batch(candidates)

        ei, mu, sigma = self.expected_improvement(X)
        best_idx = np.argmax(ei)

        # Debug output
        mode = "🔎 Exploring" if sigma[best_idx] > 0.1 else "🎯 Exploiting"
        print(f"{mode} (σ={sigma[best_idx]:.3f}, μ={mu[best_idx]:.3f}, EI={ei[best_idx]:.4f})")

        return candidates[best_idx]

    def run(self, train_fn, trials=20, patience=10, parallel_eval=1):
        """
        Main optimization loop.

        Args:
            train_fn: Objective function
            trials: Number of optimization trials
            patience: Stop if no improvement for this many trials
            parallel_eval: Number of configs to evaluate in parallel per trial
        """
        best = {"metrics": float("-inf")}
        no_improvement_count = 0

        for i in range(trials):
            # Fit GP on accumulated data
            X = self.encode_batch([r["config"] for r in self.results])
            y = np.array([r["metrics"] for r in self.results])
            self.gp.fit(X, y)

            # Parallel evaluation of multiple candidates
            if parallel_eval > 1:
                configs = [self.bayesian_sample() for _ in range(parallel_eval)]

                with ThreadPoolExecutor(max_workers=parallel_eval) as executor:
                    scores = list(executor.map(train_fn, configs))

                for config, score in zip(configs, scores):
                    self.results.append({"config": config, "metrics": score})
                    if score > best["metrics"]:
                        best = {"config": config, "metrics": score}
                        no_improvement_count = 0

                trial_best = max(scores)
            else:
                config = self.bayesian_sample()
                score = train_fn(config)
                self.results.append({"config": config, "metrics": score})
                trial_best = score

                if score > best["metrics"]:
                    best = {"config": config, "metrics": score}
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            best_so_far = max(r["metrics"] for r in self.results)
            print(f"Trial {i + 1}/{trials}: score={trial_best:.4f} | best={best_so_far:.4f}")

            # Early stopping
            if no_improvement_count >= patience:
                print(f"⏹️ Early stopping: no improvement for {patience} trials")
                break

        self.save()
        return best

    def save(self, path="results.json"):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Saved {len(self.results)} results to {path}")

    def load(self, path="results.json"):
        """Load results from JSON file."""
        try:
            with open(path, 'r') as f:
                self.results = json.load(f)
            print(f"📂 Loaded {len(self.results)} results from {path}")
        except FileNotFoundError:
            print(f"⚠️ No results file found at {path}")
            self.results = []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing {path}: {e}")
            self.results = []

    def plot(self):
        """Generate convergence and exploration plots."""
        import matplotlib.pyplot as plt

        if not self.results:
            print("No results to plot!")
            return

        metrics = [r["metrics"] for r in self.results]
        trials = list(range(1, len(metrics) + 1))

        # Convergence plot
        best_so_far = np.maximum.accumulate(metrics)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(trials, best_so_far, marker='o', markersize=3, linewidth=1.5)
        axes[0].set_title("Best Score Convergence")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Best Score")
        axes[0].grid(True, alpha=0.3)

        # Exploration scatter
        axes[1].scatter(trials, metrics, alpha=0.6, s=20)
        axes[1].plot(trials, best_so_far, 'r-', linewidth=1, alpha=0.7, label='Best')
        axes[1].set_title("Score Distribution Across Trials")
        axes[1].set_xlabel("Trial")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_best(self):
        """Return the best result found so far."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r["metrics"])
