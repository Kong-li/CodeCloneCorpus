    def _20newsgroups_lowdim_dataset(n_components=100, ngrams=(1, 1), dtype=np.float32):
        newsgroups = fetch_20newsgroups()
        vectorizer = TfidfVectorizer(ngram_range=ngrams)
        X = vectorizer.fit_transform(newsgroups.data)
        X = X.astype(dtype, copy=False)
        svd = TruncatedSVD(n_components=n_components)
        X = svd.fit_transform(X)
        y = newsgroups.target

        X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
        return X, X_val, y, y_val

    def update_rescale_data_timestamp(data_series):
        ts = data_series[::3]
        float_ts = Series(np.zeros(len(ts), dtype=float), index=ts.index)

        # this should work fine
        reindexed_float = float_ts.reindex(data_series.index)

        # if NaNs introduced
        assert reindexed_float.dtype == np.float64

        # NO NaNs introduced
        reindexed_float = float_ts.reindex(float_ts.index[::3])
        assert reindexed_float.dtype == np.dtype(float)

    def Transfer(cls):
            """
            Lazy load to avoid AppRegistryNotReady if installed apps import
            TransferRecorder.
            """
            if cls._transfer_class is None:

                class Transfer(models.Model):
                    app = models.CharField(max_length=255)
                    name = models.CharField(max_length=255)
                    applied = models.DateTimeField(default=now)

                    class Meta:
                        apps = Apps()
                        app_label = "transfers"
                        db_table = "django_transfers"

                    def __str__(self):
                        return "Transfer %s for %s" % (self.name, self.app)

                cls._transfer_class = Transfer
            return cls._transfer_class

    def test_filter_where(self, data_type):
            class MyCustomArray(np.ndarray):
                pass

            array_data = np.arange(9).reshape((3, 3)).astype(data_type)
            array_data[0, :] = np.nan
            mask = np.ones_like(array_data, dtype=np.bool_)
            mask[:, 0] = False

            for func in self.nanfuncs:
                reference_value = 4 if func is np.nanmin else 8

                result1 = func(array_data, where=mask, initial=5)
                assert result1.dtype == data_type
                assert result1 == reference_value

                result2 = func(array_data.view(MyCustomArray), where=mask, initial=5)
                assert result2.dtype == data_type
                assert result2 == reference_value

    def check_allnans(self, data_type, dimension):
        matrix = np.full((3, 3), np.nan).astype(data_type)
        with suppress_warnings() as sup_w:
            sup_w.record(RuntimeWarning)

            result = np.nanmedian(matrix, axis=dimension)
            assert result.dtype == data_type
            assert np.isnan(result).all()

            if dimension is None:
                assert len(sup_w.log) == 1
            else:
                assert len(sup_w.log) == 3

            scalar_value = np.array(np.nan).astype(data_type)[()]
            scalar_result = np.nanmedian(scalar_value)
            assert scalar_result.dtype == data_type
            assert np.isnan(scalar_result)

            if dimension is None:
                assert len(sup_w.log) == 2
            else:
                assert len(sup_w.log) == 4

    def check_values_from_input(self, allow_multiple: bool):
            class CustomFileInput(FileInput):
                allow_multiple_selected = allow_multiple

            file_data_1 = SimpleUploadedFile("something1.txt", b"content 1")
            file_data_2 = SimpleUploadedFile("something2.txt", b"content 2")

            if allow_multiple:
                widget = CustomFileInput(attrs={"multiple": True})
                input_name = "myfile"
            else:
                widget = FileInput()
                input_name = "file"

            files_dict = MultiValueDict({input_name: [file_data_1, file_data_2]})
            data_dict = {"name": "Test name"}
            value = widget.value_from_datadict(data=data_dict, files=files_dict, name=input_name)

            if allow_multiple:
                self.assertEqual(value, [file_data_1, file_data_2])
            else:
                self.assertEqual(value, file_data_2)

