# Sprint 2: Universal Test Suite Manager & Validation - Living Document

> **Sprint Duration**: 2 Weeks  
> **Status**: 🟡 **Ready to Start** (Sprint 1 Complete ✅)  
> **Completion**: 0% (0/52 tasks complete)  
> **Sprint Goal**: Build comprehensive system to load, validate, and organize user-submitted Golden Sets across all ML domains

[← Back to Project Dashboard](project-status-dashboard.md)

---

## 📊 Sprint Overview

**Start Date**: TBD  
**End Date**: TBD  
**Sprint Lead**: TBD

### Sprint Dependencies
- ✅ **Sprint 1 Complete**: Database schema operational with `test_cases` table
- ✅ **Sprint 1 Complete**: ORM models implemented (`TestCase`, `ModelRun`, `Response`, `Evaluation`)
- ✅ **Sprint 1 Complete**: CRUD operations working with user-first defaults
- ✅ **Sprint 1 Complete**: Alembic migrations configured

### Sprint Objectives
By the end of this sprint, we will have:
1. ✅ Universal test suite format specification (JSON/YAML/CSV)
2. ✅ Multi-format parser that handles diverse ML domains
3. ✅ Comprehensive validation system for inputs and outputs
4. ✅ CLI command for loading test suites
5. ✅ Domain-specific validators for 5+ input types
6. ✅ Domain-specific validators for 5+ output types
7. ✅ Batch validation with detailed error reporting

---

## 🎯 Sprint Goals & Success Metrics

### Primary Goals
- [ ] **G1**: Can load test suites for Computer Vision, NLP, and Time Series
- [ ] **G2**: Validation catches 95%+ of malformed test cases
- [ ] **G3**: Clear error messages guide users to fix invalid data
- [ ] **G4**: Handles 10,000+ test cases efficiently (<30 seconds)
- [ ] **G5**: CLI is intuitive for first-time users

### Success Metrics
- ✅ Successfully loads 3+ example test suites (CV, NLP, Time Series)
- ✅ Validation rejects invalid test cases with clear error messages
- ✅ Can parse JSON, YAML, and CSV formats
- ✅ Loading 1000 test cases completes in <10 seconds
- ✅ 90%+ test coverage for validation logic
- ✅ Documentation covers all supported formats

---

## 📋 Task Breakdown

### 1. Test Suite Format Specification (8 tasks)

#### 1.1 Design Universal Format

**Goal**: Define a flexible, domain-agnostic test suite format that works for all ML types

- [ ] **T1.1.1**: Define core test suite schema (metadata + test cases)
- [ ] **T1.1.2**: Document Computer Vision test case format
- [ ] **T1.1.3**: Document NLP test case format
- [ ] **T1.1.4**: Document Time Series test case format
- [ ] **T1.1.5**: Document Recommender System test case format
- [ ] **T1.1.6**: Document Tabular ML test case format
- [ ] **T1.1.7**: Create JSON Schema for validation
- [ ] **T1.1.8**: Write format specification document

**Assignee**: TBD  
**Status**: 🔴 Not Started  
**Estimated Hours**: 6h  
**Priority**: 🔥 Critical

**Universal Test Suite Format** (`docs/test_suite_format.md`):

```yaml
# Universal Test Suite Format Specification

## Core Structure

test_suite:
  metadata:
    name: string                    # Suite name
    description: string             # What this suite tests
    model_type: string              # 'computer_vision', 'nlp', 'time_series', etc.
    version: string                 # Suite version (e.g., "1.0.0")
    created_by: string              # Author
    created_at: datetime            # Creation timestamp
    tags: list[string]              # Organizational tags
    
  test_cases: list[TestCase]        # List of test cases

## Test Case Structure

TestCase:
  # Input specification
  input_data: dict                  # Flexible JSONB storage
  input_type: string                # Type hint for preprocessing
  input_format: string (optional)   # Format details
  
  # Expected output
  ground_truth: dict                # Expected output (JSONB)
  output_type: string               # Type hint for evaluation
  
  # Organization
  category: string (optional)       # Test category
  tags: list[string] (optional)     # Test tags
  difficulty: string (optional)     # 'easy', 'medium', 'hard'
  
  # Metadata
  description: string (optional)    # Human-readable description
  metadata: dict (optional)         # Additional context

## Input Types

- text: Plain text input
- image_path: Path to image file
- image_url: URL to image
- image_base64: Base64-encoded image
- tabular: Structured data (features)
- audio_path: Path to audio file
- audio_url: URL to audio
- time_series: Sequential numerical data
- video_path: Path to video file

## Output Types

- classification: Single label prediction
- multi_label_classification: Multiple labels
- regression: Numerical value(s)
- text: Generated text
- bounding_boxes: Object detection boxes
- segmentation_mask: Pixel-wise labels
- ranking: Ordered list of items
- embedding: Vector representation
```

**Example Test Suites**:

```json
// Computer Vision - Image Classification
{
  "metadata": {
    "name": "Animal Classification Test Suite",
    "description": "Test suite for domestic animal classifier",
    "model_type": "computer_vision",
    "version": "1.0.0",
    "created_by": "data_team",
    "tags": ["animals", "classification", "production"]
  },
  "test_cases": [
    {
      "input_data": {
        "image_path": "tests/images/cat_001.jpg"
      },
      "input_type": "image_path",
      "ground_truth": {
        "label": "cat",
        "confidence_threshold": 0.8
      },
      "output_type": "classification",
      "category": "domestic_animals",
      "tags": ["cat", "easy"],
      "difficulty": "easy",
      "description": "Clear image of a tabby cat"
    },
    {
      "input_data": {
        "image_path": "tests/images/dog_001.jpg"
      },
      "input_type": "image_path",
      "ground_truth": {
        "label": "dog",
        "confidence_threshold": 0.8
      },
      "output_type": "classification",
      "category": "domestic_animals",
      "tags": ["dog", "easy"],
      "difficulty": "easy"
    }
  ]
}

// NLP - Text Generation
{
  "metadata": {
    "name": "Question Answering Test Suite",
    "model_type": "nlp",
    "version": "1.0.0"
  },
  "test_cases": [
    {
      "input_data": {
        "text": "What is the capital of France?"
      },
      "input_type": "text",
      "ground_truth": {
        "text": "Paris",
        "required_elements": ["Paris"],
        "forbidden_elements": ["London", "Berlin"]
      },
      "output_type": "text",
      "category": "geography",
      "difficulty": "easy"
    },
    {
      "input_data": {
        "text": "Explain quantum entanglement in simple terms."
      },
      "input_type": "text",
      "ground_truth": {
        "required_elements": ["particles", "connected", "distance"],
        "min_length": 50,
        "max_length": 500
      },
      "output_type": "text",
      "category": "science",
      "difficulty": "hard"
    }
  ]
}

// Time Series - Forecasting
{
  "metadata": {
    "name": "Stock Price Forecasting Test Suite",
    "model_type": "time_series",
    "version": "1.0.0"
  },
  "test_cases": [
    {
      "input_data": {
        "sequence": [100, 102, 105, 103, 108, 110, 107],
        "forecast_horizon": 3,
        "frequency": "daily"
      },
      "input_type": "time_series",
      "ground_truth": {
        "values": [109, 111, 113],
        "tolerance": 5.0
      },
      "output_type": "regression",
      "category": "stock_prices",
      "difficulty": "medium"
    }
  ]
}

// Object Detection
{
  "metadata": {
    "name": "Pedestrian Detection Test Suite",
    "model_type": "computer_vision",
    "version": "1.0.0"
  },
  "test_cases": [
    {
      "input_data": {
        "image_path": "tests/images/street_scene_001.jpg"
      },
      "input_type": "image_path",
      "ground_truth": {
        "objects": [
          {
            "bbox": [120, 80, 200, 350],
            "class": "person",
            "confidence_threshold": 0.7
          },
          {
            "bbox": [400, 100, 480, 340],
            "class": "person",
            "confidence_threshold": 0.7
          }
        ],
        "min_iou": 0.5
      },
      "output_type": "bounding_boxes",
      "category": "pedestrian_detection",
      "difficulty": "medium"
    }
  ]
}

// Recommender System
{
  "metadata": {
    "name": "Movie Recommendation Test Suite",
    "model_type": "recommender",
    "version": "1.0.0"
  },
  "test_cases": [
    {
      "input_data": {
        "user_id": "user_12345",
        "context": {
          "recently_watched": [101, 205, 308],
          "genres_liked": ["action", "sci-fi"]
        },
        "k": 10
      },
      "input_type": "tabular",
      "ground_truth": {
        "relevant_items": [401, 502, 603, 704],
        "min_relevant_in_top_k": 2
      },
      "output_type": "ranking",
      "category": "cold_start",
      "difficulty": "hard"
    }
  ]
}
```

---

### 2. Multi-Format Parser Implementation (10 tasks)

#### 2.1 Core Parser Infrastructure

- [ ] **T2.1.1**: Create `TestSuiteParser` base class
- [ ] **T2.1.2**: Implement JSON parser
- [ ] **T2.1.3**: Implement YAML parser
- [ ] **T2.1.4**: Implement CSV parser (for simple cases)
- [ ] **T2.1.5**: Add format auto-detection (by file extension)
- [ ] **T2.1.6**: Implement parsing error handling
- [ ] **T2.1.7**: Add support for loading from file path
- [ ] **T2.1.8**: Add support for loading from string
- [ ] **T2.1.9**: Add support for loading from URL
- [ ] **T2.1.10**: Create unified parser interface

**Assignee**: TBD  
**Status**: 🔴 Not Started  
**Estimated Hours**: 8h  
**Priority**: 🔥 Critical

**Code** (`ml_eval/test_suite/parser.py`):

```python
"""Test suite parsers for multiple formats."""

import json
import yaml
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Union
from dataclasses import dataclass
import requests


@dataclass
class ParseResult:
    """Result of parsing operation."""
    success: bool
    data: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None


class TestSuiteParser(ABC):
    """Base class for test suite parsers."""
    
    @abstractmethod
    def parse(self, source: Union[str, Path]) -> ParseResult:
        """Parse test suite from source."""
        pass
    
    def _validate_structure(self, data: Dict) -> List[str]:
        """Validate basic structure of parsed data."""
        errors = []
        
        if 'metadata' not in data:
            errors.append("Missing required field: 'metadata'")
        elif not isinstance(data['metadata'], dict):
            errors.append("Field 'metadata' must be a dictionary")
        
        if 'test_cases' not in data:
            errors.append("Missing required field: 'test_cases'")
        elif not isinstance(data['test_cases'], list):
            errors.append("Field 'test_cases' must be a list")
        elif len(data['test_cases']) == 0:
            errors.append("Field 'test_cases' cannot be empty")
        
        return errors


class JSONParser(TestSuiteParser):
    """Parse JSON test suites."""
    
    def parse(self, source: Union[str, Path]) -> ParseResult:
        """Parse JSON file or string."""
        try:
            # Try to parse as file path first
            if isinstance(source, (str, Path)) and Path(source).exists():
                with open(source, 'r') as f:
                    data = json.load(f)
            else:
                # Try to parse as JSON string
                data = json.loads(source)
            
            # Validate structure
            errors = self._validate_structure(data)
            if errors:
                return ParseResult(success=False, errors=errors)
            
            return ParseResult(success=True, data=data)
            
        except json.JSONDecodeError as e:
            return ParseResult(
                success=False,
                errors=[f"Invalid JSON: {str(e)}"]
            )
        except Exception as e:
            return ParseResult(
                success=False,
                errors=[f"Error parsing JSON: {str(e)}"]
            )


class YAMLParser(TestSuiteParser):
    """Parse YAML test suites."""
    
    def parse(self, source: Union[str, Path]) -> ParseResult:
        """Parse YAML file or string."""
        try:
            # Try to parse as file path first
            if isinstance(source, (str, Path)) and Path(source).exists():
                with open(source, 'r') as f:
                    data = yaml.safe_load(f)
            else:
                # Try to parse as YAML string
                data = yaml.safe_load(source)
            
            # Validate structure
            errors = self._validate_structure(data)
            if errors:
                return ParseResult(success=False, errors=errors)
            
            return ParseResult(success=True, data=data)
            
        except yaml.YAMLError as e:
            return ParseResult(
                success=False,
                errors=[f"Invalid YAML: {str(e)}"]
            )
        except Exception as e:
            return ParseResult(
                success=False,
                errors=[f"Error parsing YAML: {str(