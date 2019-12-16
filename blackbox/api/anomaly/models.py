from flask_restplus.model import Model
from flask_restplus import fields

# API Models
new_entity_model = Model(
    "new_entity",
    {
        "attrs": fields.List(
            fields.String(),
            description="New entity attributes expected to train the models.",
            required=True,
        )
    },
)

update_entity_model = Model(
    "model",
    {
        "model_path": fields.String(description="Path of the model", required=False),
        "train_data_path": fields.String(
            description="Path of the training data", required=False
        ),
    },
)

update_entity_models = Model(
    "models", {"model": fields.Nested(update_entity_model)}
)

update_entity = Model(
    "update_entity",
    {
        "new_entity_id": fields.String(description="New entity id", required=False),
        "default": fields.String(description="New default model", required=False),
        "attrs": fields.List(
            fields.String(), description="New entity attributes", required=False
        ),
        "models": fields.Nested(update_entity_model),
    },
)