from typing import Union

from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
# Use a pipeline as a high-level helper
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"
app = FastAPI()
# a) Get predictions


# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

origins = [
    "http://localhost:4200",  # Ejemplo: http://localhost (URL de origen permitida)
    "http://example.com",  # Ejemplo: http://example.com (URL de origen permitida)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"], # Puedes especificar los métodos permitidos, por ejemplo ["GET", "POST"]
    allow_headers=["*"],  # Puedes especificar los encabezados permitidos, por ejemplo ["Authorization"]
)


class Message(BaseModel):
    message: str





@app.post("/predict/")
def generateMessage(message: Message):
    print(message.message)
    mensaje = message.message
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': mensaje  ,
        'context': '''
            Estoy interesado en obtener información sobre una ferretería llamada 'Ferretería ABC'. Esta ferretería ha estado en funcionamiento en la ciudad de [nombre de la ciudad] durante más de 20 años. Está ubicada en [dirección exacta] y es conocida por ser un lugar confiable para comprar herramientas y suministros de construcción. Me gustaría que el LLM proporcionara detalles sobre los productos que ofrece la ferretería, cómo ha evolucionado a lo largo de los años, si tiene una especialización en algún tipo de producto, cómo se diferencia de otras ferreterías de la zona y cualquier otra información relevante sobre su funcionamiento y servicios. Además, si es posible, me gustaría obtener consejos sobre cómo elegir las herramientas adecuadas para proyectos de bricolaje y renovación del hogar. Por favor, proporcione información detallada sobre 'Ferretería ABC' y sus productos y servicios.
        '''
    }
    res = nlp(QA_input)
    return res

@app.get("/")
def read_root():
    return {"Hello": "World"}



