# 🚀 AutoTuneAI - Bayesian Hyperparameter Optimization Framework

A custom Bayesian Optimization framework that automatically finds optimal hyperparameters using Gaussian Process Regression and Expected Improvement, reducing the need for manual tuning and exhaustive search methods.

---

## 🌟 Features

- **Bayesian Optimization Engine:** Uses Gaussian Process Regression (GPR) with a Matern Kernel to model the search space and predict promising configurations.

- **Expected Improvement Acquisition Function:** Balances exploration and exploitation to efficiently search for better hyperparameter combinations.

- **Parallel Evaluation:** Supports multi-threaded warm-up sampling and parallel candidate evaluation using `ThreadPoolExecutor` for faster optimization.

- **Experiment Tracking:** Automatically stores all trial results in JSON format for reproducibility and analysis.

- **Visualization Tools:** Generates convergence and score distribution plots to monitor optimization performance.

- **Early Stopping:** Stops the search automatically when no significant improvement is observed for a specified number of trials.

---

## 🛠️ Installation & Setup

### Clone the repository

```bash
git clone https://github.com/username/AutoTuneAI.git
```

### Navigate to the project directory

```bash
cd AutoTuneAI
```

### Create a virtual environment (recommended)

```bash
python -m venv venv
```

### Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Define a search space

```python
search_space = {
    "learning_rate": (0.0001, 0.1),
    "batch_size": [16, 32, 64, 128],
    "dropout": (0.1, 0.5)
}
```

### Create a training function

```python
def train_fn(config):
    # Train model and return evaluation score
    return score
```

### Run optimization

```python
from autotune import BayesianAutoTuneAgent

agent = BayesianAutoTuneAgent(
    search_space=search_space,
    train_fn=train_fn,
    warmup_samples=100
)

best_result = agent.run(
    train_fn=train_fn,
    trials=50,
    patience=10
)

print(best_result)
```

### Generate performance plots

```python
agent.plot()
```

---

## ⚙️ Configuration

The framework can be configured through initialization parameters:

| Parameter | Description | Default |
|------------|------------|----------|
| `warmup_samples` | Number of random samples collected before Bayesian optimization begins | `400` |
| `trials` | Maximum optimization iterations | `20` |
| `patience` | Early stopping threshold | `10` |
| `parallel_eval` | Number of configurations evaluated simultaneously | `1` |
| `max_workers` | Maximum threads for parallel processing | `None` |

### Example

```python
agent = BayesianAutoTuneAgent(
    search_space=search_space,
    warmup_samples=200,
    max_workers=8
)
```

---

## 📊 Technologies Used

- Python
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Concurrent Futures
- JSON

---

## 🧠 How It Works

1. Randomly samples configurations during the warm-up phase.
2. Trains a Gaussian Process surrogate model on collected results.
3. Uses the Expected Improvement acquisition function to identify promising configurations.
4. Evaluates selected configurations and updates the surrogate model.
5. Repeats until the trial limit or early stopping condition is reached.

---

## 📁 Project Structure

```text
AutoTuneAI/
│
├── autotune.py
├── results.json
├── examples/
│   └── example_usage.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the Project.
2. Create your Feature Branch.

```bash
git checkout -b feature/AmazingFeature
```

3. Commit your Changes.

```bash
git commit -m "Add some AmazingFeature"
```

4. Push to the Branch.

```bash
git push origin feature/AmazingFeature
```

5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License. See the `LICENSE` file for more information.

---

## ⭐ Future Improvements

- Support for categorical parameter encoding.
- Multi-objective optimization.
- Distributed optimization across multiple machines.
- Integration with PyTorch and TensorFlow training pipelines.
- Web dashboard for experiment monitoring.

---

Developed to make hyperparameter tuning smarter, faster, and more efficient using Bayesian Optimization.
