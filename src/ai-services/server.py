from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict

from models import mBARTWrapper

app = FastAPI(title="Yegatu Digital web services backend")

supported_tasks = ['translation', 'autocorrect', 'nextword']

# --- Input model ---
class TaskRequest(BaseModel):
    task: str = Field(..., description=f"Type of task: {','.join(supported_tasks)}")
    sentence: str = Field(..., description="Input text")
    options: Optional[Dict[str, str]] = Field(default=None, description="Extra options for translation")

# --- Output model ---
class TaskResponse(BaseModel):
    task: str
    sentence: str
    options: Optional[Dict[str, str]] = Field(default=None, description="Extra options for translation")
    output: str

# Load model
model = mBARTWrapper()

@app.post("/run_task", response_model=TaskResponse)
def run_task(request: TaskRequest):
    task = request.task.lower()

    if task not in supported_tasks:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

    kwargs = {}
    # --- translation task, populate arguments ---
    if task == "translation":
        if not request.options or "src_lang" not in request.options or "tgt_lang" not in request.options:
            raise HTTPException(status_code=400, detail="Missing src_lang or tgt_lang for translation")

        if task == 'translation':
            kwargs['src_lang'] = request.options["src_lang"]
            kwargs['tgt_lang'] = request.options["tgt_lang"]

    output = model.generate(request.sentence, task, kwargs)

    return TaskResponse(
        task=request.task,
        sentence=request.sentence,
        options=request.options,
        output=output
    )