# ML Model Evaluation Framework - Complete Sprint Plan

**Project Mission**: Build a universal, database-backed evaluation framework for testing ANY ML/AI model across all domains with user-submitted Golden Sets as the foundation.

---

## 🎯 Project Vision

Create a model-agnostic testing platform where:
1. **Practitioners submit Golden Sets** (verified test cases with inputs + expected outputs) for ANY ML model
2. Framework queries the model and captures outputs in flexible JSONB storage
3. All data stored in PostgreSQL with hybrid SQL/NoSQL architecture
4. Pluggable evaluators assess outputs using domain-appropriate metrics
5. Optional AI-powered test generation augments human-created Golden Sets (Sprint 7+)
6. Reports show pass/fail status and detect regressions across versions

**Key Differentiator**: Universal schema supporting computer vision, NLP, time series, recommenders, speech - with human-verified data as the gold standard.

---

## 📅 Sprint Breakdown (8 Sprints × 2 Weeks = 16 Weeks)

### **Sprint 1: Universal Database Schema & User-First Infrastructure** (Weeks 1-2)

**Goals**: Establish PostgreSQL database with universal, user-first schema prioritizing human-submitted Golden Sets

**Critical Philosophy**: 
- `origin='human'` is the DEFAULT for all test cases
- User-submitted data is treated as verified truth (`is_verified=TRUE`)
- AI-generated tests are an optional augmentation feature (Sprint 7)

**Tasks**:
- [ ] **Environment Setup** (5 tasks)
  - Create GitHub repository and project structure
  - Set up Python virtual environment
  - Install PostgreSQL locally
  - Install dependencies (SQLAlchemy, psycopg2, Alembic, pytest)
  - Configure `.env` file with database credentials

- [ ] **Universal Schema Design** (8 tasks)
  - Design complete universal schema with JSONB
  - Create `test_cases` table (NOT `test_prompts`)
    - `input_data JSONB` - flexible input storage
    - `ground_truth JSONB` - expected output
    - `input_type VARCHAR(50)` - 'text', 'image_path', 'tabular', 'audio_path'
    - `output_type VARCHAR(50)` - 'classification', 'text', 'regression', 'bounding_boxes'
    - `origin VARCHAR(50) DEFAULT 'human'` - **CRITICAL: human vs ai-generated**
    - `is_verified BOOLEAN DEFAULT TRUE` - assumes user data is trusted
    - `model_type VARCHAR(100)` - 'computer_vision', 'nlp', 'time_series', etc.
  - Create `model_runs` table with flexible `config JSONB`
  - Create `responses` table with `output_data JSONB`
  - Create `evaluations` table
  - Define foreign key constraints and CASCADE rules
  - Create performance indexes
  - Write `scripts/schema.sql`

- [ ] **SQLAlchemy ORM Models** (10 tasks)
  - Create database connection module with pooling
  - Create `TestCase` model (replaces `TestPrompt`)
    - Map all JSONB fields correctly
    - Include `origin` and `is_verified` fields
  - Create `ModelRun` model with flexible config
  - Create `Response` model with `output_data` JSONB
  - Create `Evaluation` model
  - Define all relationships (one-to-many, cascades)
  - Implement `__repr__` methods
  - Add validation constraints

- [ ] **Alembic Migration System** (6 tasks)
  - Initialize Alembic
  - Configure `alembic.ini` with connection string
  - Create initial migration for universal schema
  - Test migration up (apply)
  - Test migration down (rollback)
  - Document migration workflow

- [ ] **CRUD Operations - User-First** (13 tasks)
  - **TestCase CRUD**:
    - `create_test_case()` - default `origin='human'`, `is_verified=TRUE`
    - `get_test_case(case_id)` 
    - `get_test_cases()` - with pagination
    - `filter_test_cases(model_type, origin='human', tags)` - **default to human data**
    - `update_test_case()`
    - `delete_test_case()` - respects CASCADE
    - `bulk_create_test_cases()` - **explicitly sets `origin='human'`**
  - **ModelRun CRUD**:
    - `create_model_run()` - accepts `model_type` and flexible `config`
    - `get_model_run(run_id)`
    - `update_model_run_status()`
  - **Response CRUD**:
    - `create_response()` - accepts `output_data` dict for JSONB
    - `get_responses_by_run()`
  - **Evaluation CRUD**:
    - `create_evaluation()`
    - `get_evaluations_by_response()`

- [ ] **Test Data Seeding** (NEW - Critical for validation)
  - Create `scripts/seed_user_data.py`
  - Define 10 mixed-domain test cases (human-submitted):
    - 3 Computer Vision (classification + object detection)
    - 3 NLP (text generation + sentiment)
    - 2 Time Series (forecasting)
    - 2 Recommender Systems (ranking)
  - All with `origin='human'`, `is_verified=TRUE`
  - Validate JSONB storage works across all domains

**Deliverables**:
- ✅ Universal database schema (PostgreSQL + JSONB)
- ✅ Complete ORM models with `TestCase` (not `TestPrompt`)
- ✅ CRUD operations supporting user-submitted Golden Sets
- ✅ Alembic migrations
- ✅ Seed script demonstrating multi-domain data loading
- ✅ 90%+ test coverage for database module

**Success Metrics**:
- Can store test cases for CV, NLP, time series models
- JSONB storage handles diverse data formats
- Can filter by `origin='human'` to retrieve only Golden Sets
- Foreign key relationships and cascades work correctly
- Queries execute in <100ms

---

### **Sprint 2: Universal Test Suite Manager & Validation** (Weeks 3-4)

**Goals**: Build system to load, validate, and organize user-submitted Golden Sets

**Tasks**:
- [ ] Create universal test suite format specification (JSON/YAML)
- [ ] Implement multi-format parser (JSON, YAML, CSV)
- [ ] Build validation logic for different input/output types
- [ ] Create `TestSuiteManager` class with model type detection
- [ ] Implement input data validators:
  - [ ] Text input validator
  - [ ] Image path validator (file exists, valid extensions)
  - [ ] Tabular input validator (schema validation)
  - [ ] Audio path validator
  - [ ] Time series sequence validator
- [ ] Implement output validators:
  - [ ] Classification output (label, confidence)
  - [ ] Regression output (numerical values)
  - [ ] Bounding box output (coordinates, classes)
  - [ ] Text output (string validation)
  - [ ] Ranking output (item lists, scores)
- [ ] Add test suite versioning and metadata
- [ ] Build filtering API (by model type, category, tags, origin)
- [ ] Create CLI command: `ml-eval load-suite`
- [ ] Add duplicate detection
- [ ] Implement batch validation with error reporting

**Deliverables**:
- Universal test suite parser supporting 5+ data types
- Validation system with domain-specific rules
- CLI for loading user-submitted test suites
- Comprehensive error messages for invalid data

**Success Metrics**:
- Can load test suites for CV, NLP, and time series
- Validates input/output formats correctly (95%+ accuracy)
- Handles 10,000+ test cases efficiently
- Clear error messages guide users to fix invalid data

---

### **Sprint 3: Universal Model Query Engine** (Weeks 5-6)

**Goals**: Build system to query ANY model type and capture outputs in JSONB

**Tasks**:
- [ ] Design universal model interface abstraction
- [ ] Implement API model adapter (REST, gRPC)
- [ ] Implement local model adapter (PyTorch, TF, sklearn)
- [ ] Build input preprocessors:
  - [ ] Image preprocessor (load from path, resize, normalize)
  - [ ] Text preprocessor (tokenization, encoding)
  - [ ] Tabular preprocessor (feature scaling, encoding)
  - [ ] Audio preprocessor (load, resample)
  - [ ] Time series preprocessor
- [ ] Build output parsers for different formats:
  - [ ] Parse classification responses (extract label/confidence)
  - [ ] Parse bounding box responses (extract boxes/classes)
  - [ ] Parse text generation responses
  - [ ] Parse regression outputs
- [ ] Add parallel execution with ThreadPoolExecutor
- [ ] Implement rate limiting for API endpoints
- [ ] Add latency measurement (per query)
- [ ] Add memory measurement (optional)
- [ ] Create progress tracking for long runs
- [ ] Handle query failures gracefully (store error messages)
- [ ] Store all outputs in `output_data JSONB`

**Deliverables**:
- Universal model query engine supporting 5+ input types
- Input preprocessors for each domain
- Output parsers storing results in JSONB
- Parallel execution with configurable workers
- Robust error handling

**Success Metrics**:
- Can query CV, NLP, and time series models
- Preprocesses inputs correctly for each domain (100% accuracy)
- Handles 1000+ queries in parallel
- Accurately measures latency per query (<5% error)
- All outputs stored in flexible JSONB format

---

### **Sprint 4: Response Storage & Universal Output Handling** (Weeks 7-8)

**Goals**: Store outputs from any model type with full traceability to input test cases

**Tasks**:
- [ ] Implement universal response storage with JSONB
- [ ] Build output post-processors:
  - [ ] Normalize classification outputs
  - [ ] Normalize bounding box coordinates
  - [ ] Extract key metrics from text outputs
- [ ] Add response-to-test-case association tracking
- [ ] Implement duplicate detection (same run + test case)
- [ ] Create response retrieval API with joins:
  - [ ] Get response with original test case data
  - [ ] Get all responses for a run
  - [ ] Filter by model type, category, origin
- [ ] Implement pagination for large result sets
- [ ] Build response export functionality (JSON, CSV)
- [ ] Create data integrity checks (foreign keys valid)
- [ ] Add response comparison utilities (for regression detection)

**Deliverables**:
- Universal response storage in JSONB
- Query API with domain-specific filters
- Export functionality
- Data integrity validation

**Success Metrics**:
- Stores outputs from CV, NLP, time series models
- 100% accurate test case-response association
- Retrieves 10,000 responses in <1 second
- Exports work for all output types

---

### **Sprint 5: Pluggable Evaluator System - Part 1** (Weeks 9-10)

**Goals**: Build core evaluators for different ML domains, comparing model outputs to user-submitted ground truth

**Tasks**:
- [ ] Create `BaseEvaluator` abstract class
- [ ] Implement classification evaluators:
  - [ ] `ExactMatchEvaluator` - label matches ground truth
  - [ ] `TopKAccuracyEvaluator` - top-K predictions
  - [ ] `ConfusionMatrixEvaluator` - full confusion matrix
- [ ] Implement regression evaluators:
  - [ ] `MSEEvaluator` - Mean Squared Error
  - [ ] `MAEEvaluator` - Mean Absolute Error
  - [ ] `R2ScoreEvaluator` - R² score
  - [ ] `MAPEEvaluator` - Mean Absolute Percentage Error
- [ ] Implement text evaluators:
  - [ ] `ExactMatchEvaluator` (for text)
  - [ ] `BLEUEvaluator` - BLEU score
  - [ ] `ROUGEEvaluator` - ROUGE-L score
  - [ ] `ContainsKeywordsEvaluator` - required elements check
- [ ] Add evaluation result storage
- [ ] Create evaluation metrics calculation
- [ ] Build configurable thresholds per evaluator
- [ ] Implement pass/fail determination logic

**Deliverables**:
- 11+ working evaluators across 3 domains
- Evaluation storage in database
- Configurable thresholds

**Success Metrics**:
- Evaluators produce normalized scores (0-1)
- Can evaluate 1000 responses in <30 seconds
- Metrics align with standard implementations (sklearn, NLTK)

---

### **Sprint 6: Pluggable Evaluator System - Part 2** (Weeks 11-12)

**Goals**: Add advanced evaluators and plugin system

**Tasks**:
- [ ] Implement computer vision evaluators:
  - [ ] `IoUEvaluator` - Intersection over Union
  - [ ] `mAPEvaluator` - mean Average Precision
  - [ ] `PrecisionRecallEvaluator` - PR curves
  - [ ] `PixelAccuracyEvaluator` (segmentation)
- [ ] Implement ranking/recommender evaluators:
  - [ ] `PrecisionAtKEvaluator` - Precision@K
  - [ ] `NDCGEvaluator` - Normalized Discounted Cumulative Gain
  - [ ] `MAPEvaluator` - Mean Average Precision
  - [ ] `RecallAtKEvaluator` - Recall@K
- [ ] Implement semantic similarity evaluators:
  - [ ] `SemanticSimilarityEvaluator` - embedding-based
  - [ ] `BERTScoreEvaluator` - contextualized similarity
- [ ] Build custom evaluator plugin system:
  - [ ] Define plugin interface
  - [ ] Implement plugin registration
  - [ ] Create example custom evaluator
- [ ] Create evaluator comparison view (side-by-side metrics)
- [ ] Add multi-criteria evaluation support
- [ ] Implement evaluator result caching

**Deliverables**:
- 10+ additional evaluators (21+ total)
- Plugin system for custom evaluators
- Multi-criteria evaluation framework

**Success Metrics**:
- Covers 6+ ML domains comprehensively
- Plugin system allows easy extensibility
- Custom evaluators work seamlessly

---

### **Sprint 7: AI-Powered Test Generation (Optional Augmentation)** (Weeks 13-14)

**Goals**: Build automatic test case generation to AUGMENT user-submitted Golden Sets

**Critical Philosophy**:
- AI-generated tests have `origin='ai-generated'`, `is_verified=FALSE`
- Users can promote AI tests to Golden Sets after manual verification
- Golden Sets remain the default for all evaluations

**Tasks**:
- [ ] Design test generation API
- [ ] Implement `TestCaseGenerator` class
- [ ] Integrate with Anthropic Claude API
- [ ] Integrate with OpenAI GPT API
- [ ] Create generation prompt templates
- [ ] Implement generation strategies:
  - [ ] Basic test generation (happy path)
  - [ ] Edge case generation
  - [ ] Adversarial test generation
  - [ ] Coverage-based generation
- [ ] Build domain-specific generators:
  - [ ] CV test generator (synthetic descriptions)
  - [ ] NLP test generator (prompt diversity)
  - [ ] Time series test generator
- [ ] Add validation for generated tests
- [ ] Create CLI: `ml-eval generate-tests`
- [ ] Implement cost tracking (API usage)
- [ ] Add verification workflow (promote to Golden Set)

**Deliverables**:
- Working test generation system
- Integration with Claude and GPT
- Domain-specific generation strategies
- Verification workflow for promoting AI tests

**Success Metrics**:
- Can generate 100 test cases in <2 minutes
- Generated tests are diverse and realistic
- 90%+ of generated tests are valid
- Cost per 100 tests < $1

---

### **Sprint 8: Reporting, Comparison & Production Polish** (Weeks 15-16)

**Goals**: Build comprehensive reporting, version comparison, and production readiness

**Tasks**:
- [ ] Create universal report aggregation logic
- [ ] Implement domain-specific report sections:
  - [ ] Classification reports (accuracy, confusion matrix)
  - [ ] Regression reports (MSE, MAE, R², error distribution)
  - [ ] Ranking reports (Precision@K, NDCG curves)
  - [ ] Object detection reports (mAP, PR curves)
- [ ] Build model version comparison engine
- [ ] Implement regression detection across versions:
  - [ ] Identify test cases with score drops
  - [ ] Flag new failures
  - [ ] Highlight improved cases
- [ ] Create report generation (HTML, JSON, PDF)
- [ ] Add domain-specific visualizations:
  - [ ] Confusion matrices (Plotly)
  - [ ] PR curves
  - [ ] Error distributions
  - [ ] Score histograms
- [ ] Build complete CLI interface
- [ ] Create REST API with FastAPI:
  - [ ] Endpoints for loading test suites
  - [ ] Endpoints for running evaluations
  - [ ] Endpoints for retrieving results
- [ ] Add authentication/authorization (JWT)
- [ ] Implement comprehensive documentation:
  - [ ] API documentation (OpenAPI)
  - [ ] User guide
  - [ ] Developer guide
- [ ] Create example test suites (5+ domains)
- [ ] Add tutorial notebooks (Jupyter)
- [ ] Create Docker deployment (docker-compose)
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [ ] Performance optimization (query indexes, caching)
- [ ] Security audit
- [ ] Load testing (1000+ concurrent evaluations)

**Deliverables**:
- Comprehensive reporting system
- Version comparison with regression detection
- Full CLI and REST API
- Complete documentation
- Docker deployment
- Production-ready system

**Success Metrics**:
- Reports generate in <5 seconds for 1000 test cases
- Accurately detects regressions between versions
- New users can run first evaluation in <15 minutes
- API response time <200ms
- Documentation covers all features
- System scales to 100k+ test cases

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────┐
│         CLI / REST API / Web UI                 │
│  (Universal interface for all model types)      │
└─────────────────┬───────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ↓             ↓             ↓
┌──────────┐ ┌──────────┐ ┌──────────┐
│Test Suite│ │Test Case │ │ Model    │
│ Manager  │ │Generator │ │ Registry │
│          │ │(AI-opt.) │ │          │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     └────────────┼────────────┘
                  │
                  ↓
     ┌────────────────────────────┐
     │  PostgreSQL + JSONB        │
     │  • test_cases (universal)  │
     │  • model_runs              │
     │  • responses (universal)   │
     │  • evaluations             │
     └────────────┬───────────────┘
                  │
                  ↓
     ┌────────────────────────────┐
     │   Model Query Engine       │
     │  • Input Preprocessors     │
     │  • Model Adapters          │
     │  • Output Parsers          │
     └────────────┬───────────────┘
                  │
                  ↓
     ┌────────────────────────────┐
     │   Evaluation Engine        │
     │  • 20+ Domain Evaluators   │
     │  • Plugin System           │
     │  • Multi-criteria Support  │
     └────────────┬───────────────┘
                  │
                  ↓
     ┌────────────────────────────┐
     │    Reporting Engine        │
     │  • Universal Reports       │
     │  • Version Comparison      │
     │  • Regression Detection    │
     └────────────────────────────┘
```

---

## ✅ Success Criteria

The project succeeds when:
1. ✅ Works for 5+ ML domains (CV, NLP, time series, recommender, speech)
2. ✅ User-submitted Golden Sets are the default data source
3. ✅ Framework evaluates 1000 test cases in <10 minutes
4. ✅ JSONB storage handles ANY data type without schema changes
5. ✅ Reports accurately identify model regressions
6. ✅ System scales to 100k+ test cases
7. ✅ 50+ practitioners across different domains actively using the system
8. ✅ Supports local models AND API endpoints

---

## 🎓 Key Technical Innovations

### 1. User-First Philosophy
Human-submitted Golden Sets (`origin='human'`) are the foundation, not AI-generated tests

### 2. Universal JSONB Storage
Hybrid SQL/NoSQL approach handles ANY data type without schema migrations

### 3. Pluggable Evaluator Architecture
Easy to add new evaluators for new domains via plugin system

### 4. Domain-Agnostic Pipeline
Same infrastructure works for vision, NLP, time series, recommenders, speech

### 5. Version Tracking at Scale
Database tracks all model versions across all domains with full history

---

## 📈 Post-Launch Features (Future Sprints 9+)

### Sprint 9: Advanced Features
- Web dashboard for visualization (React + FastAPI)
- Real-time monitoring and alerting (webhooks)
- A/B testing framework (multi-variant comparison)
- Multi-model comparison (3+ versions side-by-side)

### Sprint 10: MLOps Integration
- Integration with MLflow (experiment tracking)
- Integration with Weights & Biases
- Integration with AWS SageMaker
- Slack/email notifications

### Sprint 11: Advanced Analytics
- Failure analysis (clustering similar failures)
- Performance profiling (latency distributions)
- Cost analysis (API usage tracking)
- Data drift detection

### Sprint 12: Enterprise Features
- Multi-tenant support (org-level isolation)
- Team collaboration (shared test suites)
- Role-based access control (RBAC)
- Custom SLAs and thresholds
- Audit logging