# -*- coding: utf-8 -*-
from pipeline import IntentPipeline
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union


class Item(BaseModel):
    text: str
    threshold: Union[float, float] = 0.01


pipeline = IntentPipeline()
app = FastAPI()


@app.post("/intent")
async def intent_classification(item: Item):
    return pipeline(item.text, item.threshold)
