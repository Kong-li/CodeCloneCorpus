    def configure_layer_type_activation_order(
        self, layer_name: str, activation_type: Callable, position: int, qsetting: QConfigAny
    ) -> QConfigMapping:
        """
        Set the QConfig for layers matching a combination of the given layer name, activation type,
        and the position at which the layer appears.

        If the QConfig for an existing (layer name, activation type, position) was already set, the new QConfig
        will override the old one.
        """
        self.layer_name_activation_type_order_qsettings[
            (layer_name, activation_type, position)
        ] = qsetting
        return self

    def pre_process(
            self, component: nn.Module, inputs: Tuple[Any, ...], params: Dict[str, Any]
        ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
            if not custom_autograd_enabled():
                logger.debug("%s", self._with_fqn("PSDP::pre_process"))
            with record_function(self._with_fqn("PSDP::pre_process")):
                self.current_state = ProcessingState.PROCESSING
                self.reshard(self.reshard_async_op)
                self.wait_for_reshard()
                inputs, params = self._attach_pre_hook(inputs, params)
                return inputs, params

    def test_same_predictions_multiclass_classification(
        seed, min_samples_leaf, n_samples, max_leaf_nodes
    ):
        # Same as test_same_predictions_regression but for classification
        pytest.importorskip("lightgbm")

        rng = np.random.RandomState(seed=seed)
        n_classes = 3
        max_iter = 1
        max_bins = 255
        lr = 1

        X, y = make_classification(
            n_samples=n_samples,
            n_classes=n_classes,
            n_features=5,
            n_informative=5,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=0,
        )

        if n_samples > 255:
            # bin data and convert it to float32 so that the estimator doesn't
            # treat it as pre-binned
            X = _BinMapper(n_bins=max_bins + 1).fit_transform(X).astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

        est_sklearn = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=max_iter,
            max_bins=max_bins,
            learning_rate=lr,
            early_stopping=False,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
        )
        est_lightgbm = get_equivalent_estimator(
            est_sklearn, lib="lightgbm", n_classes=n_classes
        )

        est_lightgbm.fit(X_train, y_train)
        est_sklearn.fit(X_train, y_train)

        # We need X to be treated an numerical data, not pre-binned data.
        X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

        pred_lightgbm = est_lightgbm.predict(X_train)
        pred_sklearn = est_sklearn.predict(X_train)
        assert np.mean(pred_sklearn == pred_lightgbm) > 0.89

        proba_lightgbm = est_lightgbm.predict_proba(X_train)
        proba_sklearn = est_sklearn.predict_proba(X_train)
        # assert more than 75% of the predicted probabilities are the same up to
        # the second decimal
        assert np.mean(np.abs(proba_lightgbm - proba_sklearn) < 1e-2) > 0.75

        acc_lightgbm = accuracy_score(y_train, pred_lightgbm)
        acc_sklearn = accuracy_score(y_train, pred_sklearn)

        np.testing.assert_allclose(acc_lightgbm, acc_sklearn, rtol=0, atol=5e-2)

        if max_leaf_nodes < 10 and n_samples >= 1000:
            pred_lightgbm = est_lightgbm.predict(X_test)
            pred_sklearn = est_sklearn.predict(X_test)
            assert np.mean(pred_sklearn == pred_lightgbm) > 0.89

            proba_lightgbm = est_lightgbm.predict_proba(X_train)
            proba_sklearn = est_sklearn.predict_proba(X_train)
            # assert more than 75% of the predicted probabilities are the same up
            # to the second decimal
            assert np.mean(np.abs(proba_lightgbm - proba_sklearn) < 1e-2) > 0.75

            acc_lightgbm = accuracy_score(y_test, pred_lightgbm)
            acc_sklearn = accuracy_score(y_test, pred_sklearn)
            np.testing.assert_almost_equal(acc_lightgbm, acc_sklearn, decimal=2)

    def load_data(
        path="imdb.npz",
        num_words=None,
        skip_top=0,
        maxlen=None,
        seed=113,
        start_char=1,
        oov_char=2,
        index_from=3,
        **kwargs,
    ):
        """Loads the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

        This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment
        (positive/negative). Reviews have been preprocessed, and each review is
        encoded as a list of word indexes (integers).
        For convenience, words are indexed by overall frequency in the dataset,
        so that for instance the integer "3" encodes the 3rd most frequent word in
        the data. This allows for quick filtering operations such as:
        "only consider the top 10,000 most
        common words, but eliminate the top 20 most common words".

        As a convention, "0" does not stand for a specific word, but instead is used
        to encode the pad token.

        Args:
            path: where to cache the data (relative to `~/.keras/dataset`).
            num_words: integer or None. Words are
                ranked by how often they occur (in the training set) and only
                the `num_words` most frequent words are kept. Any less frequent word
                will appear as `oov_char` value in the sequence data. If None,
                all words are kept. Defaults to `None`.
            skip_top: skip the top N most frequently occurring words
                (which may not be informative). These words will appear as
                `oov_char` value in the dataset. When 0, no words are
                skipped. Defaults to `0`.
            maxlen: int or None. Maximum sequence length.
                Any longer sequence will be truncated. None, means no truncation.
                Defaults to `None`.
            seed: int. Seed for reproducible data shuffling.
            start_char: int. The start of a sequence will be marked with this
                character. 0 is usually the padding character. Defaults to `1`.
            oov_char: int. The out-of-vocabulary character.
                Words that were cut out because of the `num_words` or
                `skip_top` limits will be replaced with this character.
            index_from: int. Index actual words with this index and higher.

        Returns:
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

        **`x_train`, `x_test`**: lists of sequences, which are lists of indexes
          (integers). If the num_words argument was specific, the maximum
          possible index value is `num_words - 1`. If the `maxlen` argument was
          specified, the largest possible sequence length is `maxlen`.

        **`y_train`, `y_test`**: lists of integer labels (1 or 0).

        **Note**: The 'out of vocabulary' character is only used for
        words that were present in the training set but are not included
        because they're not making the `num_words` cut here.
        Words that were not seen in the training set but are in the test set
        have simply been skipped.
        """
        origin_folder = (
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        )
        path = get_file(
            fname=path,
            origin=origin_folder + "imdb.npz",
            file_hash=(  # noqa: E501
                "69664113be75683a8fe16e3ed0ab59fda8886cb3cd7ada244f7d9544e4676b9f"
            ),
        )
        with np.load(path, allow_pickle=True) as f:
            x_train, labels_train = f["x_train"], f["y_train"]
            x_test, labels_test = f["x_test"], f["y_test"]

        rng = np.random.RandomState(seed)
        indices = np.arange(len(x_train))
        rng.shuffle(indices)
        x_train = x_train[indices]
        labels_train = labels_train[indices]

        indices = np.arange(len(x_test))
        rng.shuffle(indices)
        x_test = x_test[indices]
        labels_test = labels_test[indices]

        if start_char is not None:
            x_train = [[start_char] + [w + index_from for w in x] for x in x_train]
            x_test = [[start_char] + [w + index_from for w in x] for x in x_test]
        elif index_from:
            x_train = [[w + index_from for w in x] for x in x_train]
            x_test = [[w + index_from for w in x] for x in x_test]
        else:
            x_train = [[w for w in x] for x in x_train]
            x_test = [[w for w in x] for x in x_test]

        if maxlen:
            x_train, labels_train = remove_long_seq(maxlen, x_train, labels_train)
            x_test, labels_test = remove_long_seq(maxlen, x_test, labels_test)
            if not x_train or not x_test:
                raise ValueError(
                    "After filtering for sequences shorter than maxlen="
                    f"{str(maxlen)}, no sequence was kept. Increase maxlen."
                )

        xs = x_train + x_test
        labels = np.concatenate([labels_train, labels_test])

        if not num_words:
            num_words = max(max(x) for x in xs)

        # by convention, use 2 as OOV word
        # reserve 'index_from' (=3 by default) characters:
        # 0 (padding), 1 (start), 2 (OOV)
        if oov_char is not None:
            xs = [
                [w if (skip_top <= w < num_words) else oov_char for w in x]
                for x in xs
            ]
        else:
            xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

        idx = len(x_train)
        x_train, y_train = np.array(xs[:idx], dtype="object"), labels[:idx]
        x_test, y_test = np.array(xs[idx:], dtype="object"), labels[idx:]
        return (x_train, y_train), (x_test, y_test)

    def process_benchmark(param):
        benchmark_name = param.get('benchmark_name')
        num_samples = param['num_samples']
        batch_size = param['batch_size']
        jit_compile = param['jit_compile']

        if not benchmark_name:
            for name, benchmark_fn in BENCHMARK_NAMES.items():
                benchmark_fn(num_samples, batch_size, jit_compile)
            return

        if benchmark_name not in BENCHMARK_NAMES:
            raise ValueError(
                f"Invalid benchmark name: {benchmark_name}, `benchmark_name` must "
                f"be one of {list(BENCHMARK_NAMES.keys())}"
            )
        benchmark_fn = BENCHMARK_NAMES[benchmark_name]
        benchmark_fn(num_samples, batch_size, jit_compile)

