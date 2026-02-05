-- Create all tables from d7c66d5e9ce2 migration

CREATE TABLE model_runs (
    id SERIAL NOT NULL,
    model_name VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    model_type VARCHAR NOT NULL,
    model_endpoint VARCHAR,
    config JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR,
    total_cases INTEGER,
    completed_cases INTEGER,
    failed_cases INTEGER,
    PRIMARY KEY (id)
);
CREATE INDEX ix_model_runs_id ON model_runs (id);
CREATE INDEX ix_model_runs_model_name ON model_runs (model_name);

CREATE TABLE test_cases (
    id SERIAL NOT NULL,
    test_case_name VARCHAR NOT NULL,
    model_type VARCHAR NOT NULL,
    input_type VARCHAR NOT NULL,
    output_type VARCHAR NOT NULL,
    input_data JSONB NOT NULL,
    ground_truth JSONB NOT NULL,
    category VARCHAR,
    tags VARCHAR[],
    difficulty VARCHAR,
    origin VARCHAR NOT NULL,
    is_verified BOOLEAN NOT NULL,
    test_case_metadata JSONB,
    created_by VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (id)
);
CREATE INDEX ix_test_cases_category ON test_cases (category);
CREATE INDEX ix_test_cases_id ON test_cases (id);
CREATE INDEX ix_test_cases_model_type ON test_cases (model_type);
CREATE INDEX ix_test_cases_test_case_name ON test_cases (test_case_name);

CREATE TABLE responses (
    id SERIAL NOT NULL,
    run_id INTEGER NOT NULL,
    test_case_id INTEGER NOT NULL,
    output_data JSONB NOT NULL,
    latency_ms INTEGER,
    memory_mb FLOAT,
    tokens_used INTEGER,
    error_message VARCHAR,
    created_at TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES model_runs (id),
    FOREIGN KEY (test_case_id) REFERENCES test_cases (id),
    PRIMARY KEY (id),
    CONSTRAINT uq_test_case_run UNIQUE (test_case_id, run_id)
);
CREATE INDEX ix_responses_id ON responses (id);
CREATE INDEX ix_responses_run_id ON responses (run_id);
CREATE INDEX ix_responses_test_case_id ON responses (test_case_id);

CREATE TABLE evaluations (
    id SERIAL NOT NULL,
    response_id INTEGER NOT NULL,
    evaluator_type VARCHAR NOT NULL,
    score FLOAT NOT NULL,
    passed BOOLEAN NOT NULL,
    metrics JSONB,
    feedback VARCHAR,
    evaluated_at TIMESTAMP,
    FOREIGN KEY (response_id) REFERENCES responses (id),
    PRIMARY KEY (id)
);
CREATE INDEX ix_evaluations_id ON evaluations (id);
CREATE INDEX ix_evaluations_response_id ON evaluations (response_id);
