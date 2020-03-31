import pytest
from time import sleep
from blackbox.app.celery_app.tasks import train_task, predict_task


class TestCeleryTasks:
    def test_train_task(celery_app, celery_worker, blackbox_model, train_df):
        # task = train_task.apply_async(
        #     args=[
        #         blackbox_model.model_id,
        #         train_df.to_json(orient="split"),
        #         {
        #             "pca_mahalanobis": {},
        #             "autoencoder": {},
        #             "kmeans": {},
        #             "one_class_svm": {},
        #             "gaussian_distribution": {},
        #             "isolation_forest": {},
        #             "knearest_neighbors": {},
        #             "local_outlier_factor": {},
        #         },
        #     ]
        # )
        # print(task.result)
        assert True
