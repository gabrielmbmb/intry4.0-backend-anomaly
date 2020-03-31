import pytest
from blackbox.app.api.modules.ml.models import BlackboxModel


class BlackboxModel(BlackboxModel):
    meta = {"db_alias": "blackbox_test"}


@pytest.fixture
def train_df():
    import pandas as pd

    return pd.read_csv("./tests/csv/no_anomaly_data.csv", index_col=0)


@pytest.fixture
def predict_df():
    import pandas as pd

    return pd.read_csv("./tests/csv/all_data_anomalies_included.csv", index_col=0)


@pytest.fixture(scope="module", autouse=True)
def mongo_connection():
    from mongoengine import connect, disconnect

    disconnect()
    connect("blackbox_test", alias="blackbox_test", host="mongomock://localhost")
    yield
    disconnect()


@pytest.fixture
def blackbox_model(train_df):
    from blackbox.available_models import AVAILABLE_MODELS

    new_model = BlackboxModel(
        model_id="testmodel", models=AVAILABLE_MODELS, columns=list(train_df.columns),
    )
    new_model.save()
    return new_model
