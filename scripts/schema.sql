-- -----------------------------------------------------------------------------
-- 1. Test Prompts (The Golden Set)
--    The core input data and ground truth, designed to be universal via JSONB.
-- -----------------------------------------------------------------------------
CREATE TABLE test_prompts (
    id SERIAL PRIMARY KEY,

    -- Core Data
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(100) NOT NULL,    -- e.g., 'NLP', 'CV', 'Recommender'
    
    -- Universal Storage: JSONB handles ANY input/output type
    input_data JSONB NOT NULL,       
    expected_output JSONB NOT NULL,  -- The ground truth

    -- User-First Philosophy Fields (Crucial requirement from spec)
    origin VARCHAR(50) NOT NULL DEFAULT 'human',    -- 'human' (default), 'ai_generated', 'production_log'
    is_verified BOOLEAN NOT NULL DEFAULT TRUE,      -- TRUE for human-submitted Golden Sets

    -- Metadata
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexing for fast lookups by name and domain
CREATE INDEX idx_test_prompts_name ON test_prompts (name);
CREATE INDEX idx_test_prompts_domain ON test_prompts (domain);
CREATE INDEX idx_test_prompts_origin ON test_prompts (origin);


-- -----------------------------------------------------------------------------
-- 2. Model Runs
--    Tracks a specific version of a model execution.
-- -----------------------------------------------------------------------------
CREATE TABLE model_runs (
    id SERIAL PRIMARY KEY,

    -- Model Identification
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(100) NOT NULL, 
    
    -- Metadata
    started_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMP WITHOUT TIME ZONE
);

-- Indexing for tracking regressions across versions
CREATE INDEX idx_model_runs_name_version ON model_runs (model_name, model_version);


-- -----------------------------------------------------------------------------
-- 3. Responses (Links Prompt and Model Run)
--    The specific output from a model for a specific prompt.
-- -----------------------------------------------------------------------------
CREATE TABLE responses (
    id SERIAL PRIMARY KEY,

    -- Foreign Keys
    prompt_id INTEGER NOT NULL,
    model_run_id INTEGER NOT NULL,

    -- Universal Output Storage
    output_data JSONB NOT NULL,      -- The actual model prediction

    -- Metadata
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT fk_responses_prompt
        FOREIGN KEY (prompt_id) 
        REFERENCES test_prompts (id)
        ON DELETE CASCADE,

    CONSTRAINT fk_responses_model_run
        FOREIGN KEY (model_run_id) 
        REFERENCES model_runs (id)
        ON DELETE CASCADE,
        
    -- Ensure a model only runs a prompt once per run
    UNIQUE (prompt_id, model_run_id) 
);

-- Indexing for fast retrieval of all responses for a specific model run
CREATE INDEX idx_responses_model_run_id ON responses (model_run_id);
CREATE INDEX idx_responses_prompt_id ON responses (prompt_id);


-- -----------------------------------------------------------------------------
-- 4. Evaluations
--    A score or quantitative assessment of a Response.
-- -----------------------------------------------------------------------------
CREATE TABLE evaluations (
    id SERIAL PRIMARY KEY,

    -- Foreign Key
    response_id INTEGER NOT NULL,

    -- Evaluation Data
    evaluator_name VARCHAR(255) NOT NULL, -- e.g., 'BLEU', 'Accuracy', 'Custom_Metric_V1'
    score DOUBLE PRECISION NOT NULL,      -- Using DOUBLE PRECISION for floating-point scores
    is_pass BOOLEAN NOT NULL,             -- Simple pass/fail status

    -- Metadata
    evaluated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    details JSONB,                        -- Optional: Store evaluator config, error messages, etc.

    -- Constraints
    CONSTRAINT fk_evaluations_response
        FOREIGN KEY (response_id) 
        REFERENCES responses (id)
        ON DELETE CASCADE,
        
    -- Ensure an evaluation is only applied once to a response by a specific evaluator
    UNIQUE (response_id, evaluator_name) 
);

-- Indexing for fast retrieval of evaluations by response
CREATE INDEX idx_evaluations_response_id ON evaluations (response_id);