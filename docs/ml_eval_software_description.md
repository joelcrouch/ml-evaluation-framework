# ML Model Evaluation Framework - Complete Software Description

> **Mission**: A universal, database-backed evaluation platform for systematically testing ANY ML/AI model by submitting verified test cases (Golden Sets), capturing model outputs, and comparing against ground truth.

---

## 🎯 The Problem We're Solving

**The Challenge**: ML practitioners across ALL domains need systematic ways to test their models, but currently:

| Domain | Current Testing Gap | Impact |
|--------|-------------------|--------|
| **Computer Vision** | Manual inspection of predictions on sample images | Missed edge cases, no version tracking |
| **NLP/LLMs** | Ad-hoc prompting without systematic evaluation | Regression goes undetected across versions |
| **Recommender Systems** | A/B tests in production only | No pre-deployment validation |
| **Time Series** | Statistical tests on small samples | No comprehensive test coverage |
| **Speech Recognition** | Manual listening to samples | Time-intensive, not scalable |

**The Core Issue**: No universal testing framework that works for all ML domains with persistent storage and version tracking.

---

## 💡 Our Solution

A **model-agnostic test harness** where practitioners:

1. **Submit Golden Sets** (verified test cases: inputs + expected outputs)
2. **Framework queries the model** with each input
3. **Framework captures all outputs** in flexible storage
4. **Framework evaluates** outputs against ground truth
5. **Framework tracks everything** in a database for version comparison

### The Universal Testing Pattern

**ALL ML models follow the same pattern:**

```
Input → Model → Output → Evaluate(Output, GroundTruth) → Score
```

**Examples across domains:**

| Domain | Input | Expected Output | Evaluation Method |
|--------|-------|-----------------|-------------------|
| **LLM** | "What is the capital of France?" | "Paris" | Semantic similarity |
| **Computer Vision** | `image.jpg` (cat) | `{"label": "cat"}` | Exact match |
| **Recommender** | `user_id=123` | `[item_5, item_12, item_8]` | Precision@K, NDCG |
| **Time Series** | `[1,2,3,4,5]` | `[6,7,8]` | MSE, MAE |
| **Tabular** | `{age:25, income:50k}` | `"approved"` | Accuracy |
| **Object Detection** | `image.jpg` | `[{box:[x,y,w,h], class:"dog"}]` | IoU, mAP |
| **Speech** | `audio.wav` | `"hello world"` | WER (Word Error Rate) |

**Our framework handles ALL of these with the same infrastructure.**

---

## 🏗️ Technical Architecture

### Core Technology Stack

- **Language**: Python 3.9+
- **Database**: PostgreSQL 14+ with JSONB support
- **ORM**: SQLAlchemy 2.0+
- **Migrations**: Alembic
- **API**: FastAPI (planned)
- **CLI**: Click
- **Testing**: pytest

### The Hybrid SQL/NoSQL Approach

**Why PostgreSQL + JSONB?**

The framework uses PostgreSQL's JSONB columns to achieve flexibility without sacrificing relational integrity:

```sql
-- Structured SQL for relationships
test_cases.id → responses.test_case_id (foreign key)

-- Flexible JSONB for diverse ML data
test_cases.input_data = {
  "image_path": "...",        -- Computer Vision
  "text": "...",              -- NLP
  "features": [...],          -- Tabular
  "sequence": [...]           -- Time Series
}
```

**Benefits:**
- ✅ No schema migrations needed for new ML domains
- ✅ Strong foreign key integrity for relationships
- ✅ Flexible storage for any input/output format
- ✅ JSON querying capabilities (GIN indexes)
- ✅ ACID transactions for data consistency

---

## 📊 Universal Data Model

### Core Tables

#### 1. `test_cases` - The Heart of the System

Stores user-submitted Golden Sets with flexible input/output formats:

```sql
CREATE TABLE test_cases (
    id SERIAL PRIMARY KEY,
    
    -- Flexible input storage (JSONB)
    input_data JSONB NOT NULL,
    input_type VARCHAR(50) NOT NULL,  -- 'text', 'image_path', 'tabular', etc.
    input_format VARCHAR(50),          -- 'json', 'base64', 'url', etc.
    
    -- Expected output (JSONB)
    ground_truth JSONB NOT NULL,
    output_type VARCHAR(50) NOT NULL,  -- 'classification', 'regression', 'bounding_boxes', etc.
    
    -- Organization
    model_type VARCHAR(100),           -- 'computer_vision', 'nlp', 'time_series', etc.
    category VARCHAR(100),
    tags TEXT[],
    difficulty VARCHAR(50),
    
    -- User-first philosophy (CRITICAL)
    origin VARCHAR(50) DEFAULT 'human',      -- 'human' vs 'ai-generated'
    is_verified BOOLEAN DEFAULT TRUE,        -- Assumes user data is trusted
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT check_input_not_empty CHECK (input_data IS NOT NULL),
    CONSTRAINT check_ground_truth_not_empty CHECK (ground_truth IS NOT NULL)
);
```

**Key Design Principles:**

1. **`origin='human'` is DEFAULT**: User-submitted Golden Sets are the foundation
2. **`is_verified=TRUE` for human data**: Assumes users provide trusted ground truth
3. **JSONB flexibility**: Same table stores vision, NLP, time series, etc.
4. **Type hints**: `input_type` and `output_type` guide processing pipelines

**Example Records:**

```json
// Computer Vision - Image Classification
{
  "input_data": {"image_path": "tests/images/cat_001.jpg"},
  "input_type": "image_path",
  "ground_truth": {"label": "cat", "confidence": 0.95},
  "output_type": "classification",
  "model_type": "computer_vision",
  "origin": "human",
  "is_verified": true
}

// NLP - Text Generation
{
  "input_data": {"text": "What is the capital of France?"},
  "input_type": "text",
  "ground_truth": {"text": "Paris", "required_elements": ["Paris"]},
  "output_type": "text",
  "model_type": "nlp",
  "origin": "human",
  "is_verified": true
}

// Time Series - Forecasting
{
  "input_data": {"sequence": [100, 105, 102, 108, 115], "horizon": 3},
  "input_type": "tabular",
  "ground_truth": {"values": [120, 125, 122]},
  "output_type": "regression",
  "model_type": "time_series",
  "origin": "human",
  "is_verified": true
}

// Object Detection
{
  "input_data": {"image_path": "tests/images/street.jpg"},
  "input_type": "image_path",
  "ground_truth": {
    "objects": [
      {"bbox": [10, 20, 100, 150], "class": "car", "confidence": 0.95},
      {"bbox": [200, 50, 80, 120], "class": "person", "confidence": 0.88}
    ]
  },
  "output_type": "bounding_boxes",
  "model_type": "computer_vision",
  "origin": "human",
  "is_verified": true
}
```

#### 2. `model_runs` - Evaluation Tracking

```sql
CREATE TABLE model_runs (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    model_type VARCHAR(100) NOT NULL,  -- 'computer_vision', 'nlp', etc.
    model_endpoint TEXT,                -- API URL or local path
    config JSONB DEFAULT '{}',          -- Model-specific configuration
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    total_cases INTEGER DEFAULT 0,
    completed_cases INTEGER DEFAULT 0,
    failed_cases INTEGER DEFAULT 0
);
```

#### 3. `responses` - Model Outputs

```sql
CREATE TABLE responses (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    test_case_id INTEGER NOT NULL REFERENCES test_cases(id) ON DELETE CASCADE,
    
    -- Flexible output storage (JSONB)
    output_data JSONB NOT NULL,
    
    -- Performance metrics
    latency_ms INTEGER,
    memory_mb FLOAT,
    tokens_used INTEGER,           -- For LLMs
    
    -- Error handling
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_run_test_case UNIQUE(run_id, test_case_id)
);
```

#### 4. `evaluations` - Assessment Results

```sql
CREATE TABLE evaluations (
    id SERIAL PRIMARY KEY,
    response_id INTEGER NOT NULL REFERENCES responses(id) ON DELETE CASCADE,
    evaluator_type VARCHAR(100) NOT NULL,
    score FLOAT NOT NULL CHECK (score >= 0 AND score <= 1),
    passed BOOLEAN NOT NULL,
    metrics JSONB DEFAULT '{}',    -- Task-specific metrics
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_response_evaluator UNIQUE(response_id, evaluator_type)
);
```

---

## 🔄 Complete System Workflow

### Phase 1: Golden Set Submission

**User creates test suite:**

```json
{
  "model_type": "computer_vision",
  "task": "image_classification",
  "test_cases": [
    {
      "input_data": {"image_path": "tests/images/cat_001.jpg"},
      "input_type": "image_path",
      "ground_truth": {"label": "cat", "confidence": 0.95},
      "output_type": "classification",
      "category": "animals",
      "tags": ["cat", "domestic_animal"]
    },
    // ... more test cases
  ]
}
```

**CLI command:**

```bash
ml-eval load-suite tests/cv_classification.json --model-type computer_vision
```

**What happens:**
1. Parser validates JSON structure
2. System validates each test case (image path exists, ground truth format correct)
3. CRUD operation: `bulk_create_test_cases()` with `origin='human'`, `is_verified=TRUE`
4. Database stores in `test_cases` table with JSONB
5. Returns suite ID for tracking

### Phase 2: Model Evaluation Run

**CLI command:**

```bash
ml-eval run \
  --suite-id 1 \
  --model-endpoint https://api.mymodel.com/classify \
  --model-version v2.1.0 \
  --model-type computer_vision \
  --evaluators exact_match,top_k_accuracy
```

**What happens:**

1. **Create Model Run**:
   ```python
   run = create_model_run(
       session=db,
       model_name="ImageClassifier",
       model_version="v2.1.0",
       model_type="computer_vision",
       total_cases=100
   )
   ```

2. **Query Engine Processes Each Test Case**:
   ```python
   for test_case in get_test_cases(suite_id=1):
       # Preprocess input
       if test_case.input_type == 'image_path':
           image = load_image(test_case.input_data['image_path'])
           preprocessed = resize_and_normalize(image)
       
       # Query model
       start_time = time.time()
       response = requests.post(
           model_endpoint,
           json={"image": base64.encode(preprocessed)}
       )
       latency = (time.time() - start_time) * 1000
       
       # Parse output
       output_data = {
           "label": response.json()['predicted_class'],
           "confidence": response.json()['confidence']
       }
       
       # Store response
       create_response(
           session=db,
           run_id=run.id,
           test_case_id=test_case.id,
           output_data=output_data,
           latency_ms=latency
       )
   ```

3. **Evaluation Engine Assesses Each Response**:
   ```python
   for response in get_responses_by_run(run.id):
       test_case = get_test_case(response.test_case_id)
       
       # Exact Match Evaluator
       evaluator = ExactMatchEvaluator(threshold=1.0)
       result = evaluator.evaluate(
           output=response.output_data,
           ground_truth=test_case.ground_truth
       )
       
       # Store evaluation
       create_evaluation(
           session=db,
           response_id=response.id,
           evaluator_type="exact_match",
           score=result.score,
           passed=result.passed,
           metrics=result.metrics
       )
   ```

4. **Update Run Status**:
   ```python
   update_model_run_status(
       session=db,
       run_id=run.id,
       status="completed",
       completed_cases=100,
       failed_cases=0
   )
   ```

### Phase 3: Results Analysis

**CLI command:**

```bash
ml-eval report --run-id 123 --format html
```

**Generated Report Includes:**
- Overall accuracy: 94/100 (94%)
- Confusion matrix (for classification)
- Per-category performance
- Failed test cases (with details)
- Latency distribution
- Comparison to baseline (if available)

---

## 🧩 Core Components Deep Dive

### 1. Test Suite Manager

**Responsibilities:**
- Parse test suites from JSON/YAML/CSV
- Validate input/output formats
- Detect model type automatically
- Handle batch uploads
- Manage test suite versioning

**Key Classes:**

```python
class TestSuiteManager:
    def load_suite(self, file_path: str, model_type: str) -> int:
        """Load test suite from file, return suite ID."""
        
    def validate_suite(self, suite_data: dict) -> ValidationResult:
        """Validate all test cases in suite."""
        
    def get_suite_stats(self, suite_id: int) -> dict:
        """Return statistics about test suite."""
```

### 2. Universal Query Engine

**Responsibilities:**
- Abstract different model interfaces (API, local, cloud)
- Preprocess inputs based on `input_type`
- Handle rate limiting for API endpoints
- Measure latency and resource usage
- Parse outputs into standard JSONB format
- Handle errors gracefully

**Key Classes:**

```python
class ModelAdapter(ABC):
    @abstractmethod
    def query(self, input_data: dict, input_type: str) -> dict:
        """Query model and return output."""

class APIModelAdapter(ModelAdapter):
    def query(self, input_data: dict, input_type: str) -> dict:
        # Preprocess based on input_type
        preprocessed = self.preprocess(input_data, input_type)
        
        # Call API
        response = requests.post(self.endpoint, json=preprocessed)
        
        # Parse and return
        return self.parse_output(response.json())

class LocalModelAdapter(ModelAdapter):
    def query(self, input_data: dict, input_type: str) -> dict:
        # Load model if not loaded
        if self.model is None:
            self.model = torch.load(self.model_path)
        
        # Preprocess and predict
        preprocessed = self.preprocess(input_data, input_type)
        output = self.model(preprocessed)
        
        return self.parse_output(output)
```

### 3. Pluggable Evaluator System

**Base Evaluator:**

```python
from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    
    @abstractmethod
    def evaluate(self, output: dict, ground_truth: dict) -> EvaluationResult:
        """Compare output to ground truth, return score."""
        pass

@dataclass
class EvaluationResult:
    score: float          # 0.0 to 1.0
    passed: bool          # score >= threshold
    metrics: dict         # Additional metrics
    feedback: str = ""    # Human-readable explanation
```

**Domain-Specific Evaluators:**

```python
# Classification
class ExactMatchEvaluator(BaseEvaluator):
    def evaluate(self, output: dict, ground_truth: dict) -> EvaluationResult:
        predicted_label = output.get("label")
        true_label = ground_truth.get("label")
        score = 1.0 if predicted_label == true_label else 0.0
        return EvaluationResult(
            score=score,
            passed=score >= self.threshold,
            metrics={"exact_match": score}
        )

# Regression
class MSEEvaluator(BaseEvaluator):
    def evaluate(self, output: dict, ground_truth: dict) -> EvaluationResult:
        predicted = np.array(output.get("values"))
        actual = np.array(ground_truth.get("values"))
        mse = np.mean((predicted - actual) ** 2)
        # Normalize to 0-1 (lower MSE = higher score)
        score = 1.0 / (1.0 + mse)
        return EvaluationResult(
            score=score,
            passed=mse <= self.threshold,
            metrics={"mse": float(mse), "normalized_score": score}
        )

# Object Detection
class IoUEvaluator(BaseEvaluator):
    def evaluate(self, output: dict, ground_truth: dict) -> EvaluationResult:
        pred_boxes = output.get("objects", [])
        true_boxes = ground_truth.get("objects", [])
        
        ious = []
        for pred in pred_boxes:
            for true in true_boxes:
                if pred['class'] == true['class']:
                    iou = self.compute_iou(pred['bbox'], true['bbox'])
                    ious.append(iou)
        
        avg_iou = np.mean(ious) if ious else 0.0
        return EvaluationResult(
            score=avg_iou,
            passed=avg_iou >= self.threshold,
            metrics={"mean_iou": avg_iou, "num_matches": len(ious)}
        )
```

---

## 🎯 Real-World Usage Examples

### Example 1: Computer Vision - Image Classification

```bash
# 1. User creates test suite
cat > tests/animal_classifier.json << EOF
{
  "model_type": "computer_vision",
  "task": "image_classification",
  "test_cases": [
    {
      "input_data": {"image_path": "tests/images/cat_001.jpg"},
      "input_type": "image_path",
      "ground_truth": {"label": "cat"},
      "output_type": "classification",
      "category": "domestic_animals"
    },
    {
      "input_data": {"image_path": "tests/images/dog_001.jpg"},
      "input_type": "image_path",
      "ground_truth": {"label": "dog"},
      "output_type": "classification",
      "category": "domestic_animals"
    }
  ]
}
EOF

# 2. Load suite
ml-eval load-suite tests/animal_classifier.json --model-type computer_vision
# Output: Suite loaded successfully. Suite ID: 1 (2 test cases)

# 3. Run evaluation
ml-eval run \
  --suite-id 1 \
  --model-endpoint https://api.mymodel.com/classify \
  --model-version v2.1.0 \
  --evaluators exact_match,top_k_accuracy

# 4. View results
ml-eval report --run-id 1 --format html --output report.html
```

### Example 2: NLP - Sentiment Analysis

```bash
# Load sentiment test suite
ml-eval load-suite tests/sentiment.json --model-type nlp

# Run with semantic similarity evaluator
ml-eval run \
  --suite-id 2 \
  --model-endpoint https://api.mymodel.com/sentiment \
  --evaluators exact_match,semantic_similarity

# Compare two model versions
ml-eval compare --run1 5 --run2 6 --output comparison.html
```

### Example 3: Time Series - Stock Forecasting

```bash
# Load time series test suite
ml-eval load-suite tests/stock_forecast.json --model-type time_series

# Run with regression evaluators
ml-eval run \
  --suite-id 3 \
  --model-endpoint https://api.mymodel.com/forecast \
  --evaluators mse,mae,mape

# Generate detailed report
ml-eval report --run-id 10 --include-error-distribution
```

---

## 🚀 Key Advantages

### 1. Universal Application
- **Single framework** for ALL ML domains
- No need for separate testing tools per domain
- Consistent interface across teams

### 2. User-First Philosophy
- Human-submitted Golden Sets are the foundation
- AI-generated tests are optional augmentation
- Users control what constitutes "ground truth"

### 3. Flexible Yet Structured
- JSONB handles any data format
- SQL ensures relational integrity
- No schema migrations for new domains

### 4. Version Tracking
- Every model run is stored permanently
- Easy comparison between versions
- Automatic regression detection

### 5. Scalability
- PostgreSQL handles 100k+ test cases
- Parallel query execution
- Efficient indexing for fast retrieval

### 6. Extensibility
- Plugin system for custom evaluators
- Easy to add new model adapters
- Open architecture for integrations

---

## 📈 Why This Matters

**Current State of ML Testing:**
- 🔴 Teams build custom tools per domain
- 🔴 No persistent storage of test results
- 🔴 Manual tracking of model versions
- 🔴 Regressions discovered in production

**With Our Framework:**
- ✅ Universal tool across all domains
- ✅ Database-backed test history
- ✅ Automatic version comparison
- ✅ Pre-deployment validation

**Market Impact:**
- Every computer vision team needs this
- Every NLP team needs this
- Every recommender system team needs this
- Every time series team needs this
- **This is the universal testing platform for ML**

---

## 🎓 Summary

**What the software does:**

1. **Accepts** user-submitted Golden Sets (test cases with verified ground truth)
2. **Stores** everything in PostgreSQL with JSONB flexibility
3. **Queries** ANY model (vision, NLP, time series, etc.) with appropriate preprocessing
4. **Captures** outputs in flexible JSONB storage
5. **Evaluates** using domain-appropriate metrics
6. **Tracks** everything for version comparison and regression detection
7. **Scales** to 100k+ test cases with fast queries

**Core Innovation**: Hybrid SQL/NoSQL architecture + user-first philosophy + pluggable evaluators = universal testing platform for the entire ML industry.