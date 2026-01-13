Here is a list of commands to quickly get the platform up and running for a demo:

1. **Start the database:**  
```bash
./start_db.sh
```
2. Start the FastAPI server (in a separate terminal):
```
    uvicorn ml_eval.main:app --host 0.0.0.0 --port 8000
```
3. Create a ModelRun (in a new terminal):

```
curl -X 'POST' \
  'http://localhost:8000/api/v1/runs/' \
  -H 'Content-Type: application/json' \
  -d '{
    model_name: FlowerClassifier-Demo,
    model_version: 1.0,
    model_type: image_classification,
    config: {
      model_path: models/cv_flower_classifier.keras
    }
  }'
```
(This will return a JSON response with an "id" for the new ModelRun. Use this ID in the next step.)


4. Run the evaluation:
```
/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/bin/python scripts/run_evaluation.py <your_new_run_id>
```

5.Generate the report:
```
/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/bin/python scripts/generate_report.py <your_new_run_id>
```

























