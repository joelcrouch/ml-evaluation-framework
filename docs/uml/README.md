# ML Evaluation Framework - UML Diagrams

This directory contains comprehensive UML diagrams documenting the architecture, data flow, and database schema of the ML Evaluation Framework.

## Diagrams Overview

### 1. **Sequence Diagram - Complete Flow** (`sequence_diagram_complete_flow.puml`)
Shows the end-to-end execution flow from golden dataset creation through evaluation to report generation.

**Key Phases:**
- Phase 1: Golden Dataset Creation (Training & Data Export)
- Phase 2: Database Seeding (API calls to store test cases)
- Phase 3: Model Run Creation (Registering an evaluation run)
- Phase 4: Evaluation Execution (Running predictions & scoring)
- Phase 5: Report Generation (Metrics & Visualizations)

**Use this diagram to understand:**
- The complete lifecycle of an evaluation
- How components interact in sequence
- API calls and database operations
- Method signatures and data flow between objects

---

### 2. **Class Diagram - Architecture** (`class_diagram_architecture.puml`)
Shows the class structure, interfaces, and relationships between all major components.

**Key Packages:**
- Database Models (TestPrompt, ModelRun, Response, Evaluation)
- Core Interfaces (IModelAdapter, IEvaluator)
- Model Implementations (BaselineTimeSeriesModel, ImageClassifierModel, etc.)
- Adapter Implementations (BaselineTimeSeriesAdapter, ImageClassifierAdapter, etc.)
- Evaluator Implementations (MeanSquaredErrorEvaluator, AccuracyEvaluator, etc.)
- Evaluation Engine (EvaluationEngine)
- CRUD Operations
- FastAPI Router
- Test Suite Management
- Scripts

**Use this diagram to understand:**
- Class hierarchies and inheritance
- Interface implementations (Strategy pattern)
- Composition relationships (which classes contain which)
- Method signatures for key classes
- How different model types are handled uniformly

---

### 3. **Database ER Diagram** (`database_er_diagram.puml`)
Shows the database schema with tables, columns, data types, indexes, and relationships.

**Tables:**
- `test_cases`: Golden dataset storage (TestPrompt model)
- `model_runs`: Model execution tracking (ModelRun model)
- `responses`: Model output storage (Response model)
- `evaluations`: Scoring & metrics (Evaluation model)
- `alembic_version`: Migration tracking

**Use this diagram to understand:**
- Database schema structure
- Foreign key relationships
- JSONB field usage for flexible data storage
- Unique constraints (e.g., preventing duplicate evaluations)
- Indexes for query performance

---

### 4. **Data Flow Diagram** (`data_flow_diagram.puml`)
Shows how data moves through the system from a high-level architectural perspective.

**Key Components:**
- Training & Dataset Creation
- Database Seeding
- Model Evaluation
- Reporting & Visualization
- User Interactions

**Use this diagram to understand:**
- High-level system architecture
- Data transformation pipeline
- Component interactions
- File artifacts (models, datasets, reports)

---

## Viewing the Diagrams

These diagrams are written in **PlantUML** format (`.puml` files). Here are several ways to view them:

### Option 1: Online PlantUML Viewer (Easiest)
1. Go to http://www.plantuml.com/plantuml/uml/
2. Copy the contents of any `.puml` file
3. Paste into the text area
4. View the rendered diagram
5. Download as PNG/SVG if needed

### Option 2: VS Code Extension
1. Install the "PlantUML" extension in VS Code
2. Open any `.puml` file
3. Press `Alt+D` (or `Cmd+D` on Mac) to preview
4. Or right-click → "Preview Current Diagram"

### Option 3: Command Line (requires Java & Graphviz)
```bash
# Install PlantUML
sudo apt-get install plantuml graphviz  # Ubuntu/Debian
brew install plantuml graphviz          # macOS

# Generate PNG images
plantuml docs/uml/*.puml

# This creates PNG files in the same directory:
# - sequence_diagram_complete_flow.png
# - class_diagram_architecture.png
# - database_er_diagram.png
# - data_flow_diagram.png
```

### Option 4: Online PlantUML Editor
Visit https://www.planttext.com/ and paste the diagram code.

---

## Using These Diagrams

### For New Developers
1. Start with **Data Flow Diagram** to understand the high-level architecture
2. Review **Sequence Diagram** to see how components interact
3. Study **Class Diagram** to understand code organization
4. Reference **Database ER Diagram** when working with database queries

### For Documentation
- Include rendered PNGs in documentation
- Link to these `.puml` files in architecture docs
- Use as reference when explaining system design

### For Onboarding
These diagrams serve as the "map" of the codebase:
- Show how training scripts create datasets
- Explain how seeding works
- Illustrate the evaluation pipeline
- Demonstrate database schema design

### For Planning New Features
When adding new model types or evaluators:
1. Check **Class Diagram** to see where new classes fit
2. Review **Sequence Diagram** to understand integration points
3. Update **Database ER Diagram** if schema changes are needed

---

## Maintaining These Diagrams

**When to update:**
- Adding new model types → Update Class Diagram
- Changing database schema → Update ER Diagram
- Adding new API endpoints → Update Sequence & Class Diagrams
- Refactoring evaluation flow → Update Sequence & Data Flow Diagrams

**How to update:**
1. Edit the `.puml` files directly
2. Regenerate images (if storing rendered versions)
3. Commit both `.puml` source and rendered PNGs

---

## Key Architectural Patterns

These diagrams illustrate several design patterns used in the framework:

### 1. Strategy Pattern (Adapters & Evaluators)
- **Problem:** Different model types require different prediction logic
- **Solution:** `IModelAdapter` interface with multiple implementations
- **Benefit:** EvaluationEngine is model-agnostic

### 2. Strategy Pattern (Evaluators)
- **Problem:** Different metrics for different domains (MSE vs Accuracy)
- **Solution:** `IEvaluator` interface with multiple implementations
- **Benefit:** Easy to add new evaluation metrics

### 3. Repository Pattern (CRUD Operations)
- **Problem:** Database access scattered throughout codebase
- **Solution:** Centralized CRUD operations module
- **Benefit:** Single source for all database queries

### 4. Entity-Relationship Pattern (Database Models)
- **Problem:** Complex relationships between test cases, runs, responses, evaluations
- **Solution:** Well-defined foreign key relationships
- **Benefit:** Data integrity and efficient queries

---

## Related Documentation

- **Demo Guide:** `docs/DEMO_GUIDE.md` - Step-by-step usage instructions
- **Alembic Refactor:** `docs/alembic_migration_refactor.md` - Database migration approach
- **Troubleshooting:** `docs/troubleshooting_db_errors.md` - Common issues

---

## Questions?

If these diagrams don't answer your questions:
1. Check the actual source code files referenced in the diagrams
2. Review the comprehensive exploration output in the session logs
3. Run the demo guide to see the system in action
4. Ask specific questions about components or flows

---

**Last Updated:** 2026-02-07
**Framework Version:** Sprint 4
**Diagram Format:** PlantUML
