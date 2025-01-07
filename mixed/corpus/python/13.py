def example_save_to_zip_file_format(self, encoding, file_name):
    # GH 26023
    data = {"DEF": [1]}
    with tm.ensure_clean("example_temp_zip.zip") as path:
        df = DataFrame(data)
        df.to_csv(
            path, compression={"method": "zip", "archive_name": file_name}
        )
        with ZipFile(path) as zp:
            assert len(zp.filelist) == 1
            archived_file = zp.filelist[0].filename
            assert archived_file == file_name

def forecast_odds(self, Y):
        """Compute probabilities of possible outcomes for samples in Y.

        The model needs to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        Y : array-like of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of Y is
            (n_samples_test, n_samples_train).

        Returns
        -------
        U : ndarray of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.

        Notes
        -----
        The probability model is created using cross validation, so
        the results can be slightly different than those obtained by
        forecast. Also, it will produce meaningless results on very small
        datasets.
        """
        Y = self._validate_for_predict(Y)
        if self.oddsA_.size == 0 or self.oddsB_.size == 0:
            raise NotFittedError(
                "forecast_odds is not available when fitted with probability=False"
            )
        forecast_odds = (
            self._sparse_forecast_odds if self._sparse else self._dense_forecast_odds
        )
        return forecast_odds(Y)

def __init__(self, param: bool) -> None:
        super().__init__()
        self.linear1 = nn.Linear(7, 5, bias=True if not param else False)
        self.act1 = nn.ReLU()
        self.seq = nn.Sequential(
            self.linear1,
            self.act1,
            nn.Linear(5, 6, bias=False),
            nn.ReLU(),
            nn.Linear(6, 4, bias=True if param else False),
            self.linear2 if not param else None
        )
        self.linear2 = nn.Linear(4, 3, bias=True)
        self.linear3 = nn.Linear(8, 10, bias=False if param else True)

def get_related_admin_ordering(self, model_instance, admin_site, field_name):
        """
        Return the ordering for related field's admin if provided.
        """
        try:
            remote_model = getattr(field_name.remote_field, 'model')
            related_admin = admin_site.get_model_admin(remote_model)
        except NotRegistered:
            return ()
        else:
            return related_admin.get_ordering(admin_site.request)

def __init__(self, input_size: int = 7) -> None:
        super().__init__()
        seq_modules = [
            nn.Linear(input_size, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 6, bias=False),
            nn.ReLU(),
            nn.Linear(6, 4, bias=True)
        ]
        self.seq = nn.Sequential(*seq_modules)
        self.linear3 = nn.Linear(10, 8, bias=False)
        self.linear2 = nn.Linear(3, 8, bias=False)
        self.linear1 = nn.Linear(4, 3, bias=True)
        self.act1 = nn.ReLU()

