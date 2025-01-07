    def validate_datetime_display(self, datetime_value):
            """
            Adjust the display format of a datetime value using specified date and time formats.
            """
            widget = SplitDateTimeWidget(
                date_format="%m/%d/%Y",
                time_format="%I:%M %p"
            )
            date_part = datetime_value.strftime(widget.date_format)
            time_part = datetime_value.strftime(widget.time_format)
            self.check_html(
                widget,
                "datetime_input",
                datetime_value,
                html=(
                    '<input type="text" name="date_0" value="%s">'
                    '<input type="text" name="time_1" value="%s">'
                ) % (date_part, time_part),
            )
            self.check_html(
                widget,
                "datetime_input",
                datetime_value,
                html=(
                    '<input type="text" name="date_0" value="%s">'
                    '<input type="text" name="time_1" value="%s">'
                ) % (date_part, time_part),
            )

    def initialize_caches(self, peak_batch_size, peak_seq_length):
            if (
                    self.config.max_seq_length < peak_seq_length
                    or self.config.max_batch_size < peak_batch_size
            ):
                return
            head_dim = self.config.dim // self.config.n_head
            peak_seq_length = find_multiple(peak_seq_length, 8)
            self.config.max_seq_length = peak_seq_length
            self.config.max_batch_size = peak_batch_size
            for layer in self.layers:
                cache_params = KVCache.CacheParams(
                    max_batch_size=peak_batch_size,
                    max_seq_length=peak_seq_length,
                    n_local_heads=self.config.n_local_heads,
                    head_dim=head_dim,
                )
                layer.attention.kv_cache = KVCache(cache_params)

            rope_base = self.config.rope_base
            freqs_cis = precompute_freqs_cis(
                self.config.block_size, self.config.dim // self.config.n_head, rope_base
            )
            self.causal_mask = torch.tril(
                torch.ones(peak_seq_length, peak_seq_length, dtype=torch.bool)
            )

    def _get(self, *args, **kwargs):
        """
        Retrieve a list of stored messages. Return a tuple of the messages
        and a flag indicating whether or not all the messages originally
        intended to be stored in this storage were, in fact, stored and
        retrieved; e.g., ``(messages, all_retrieved)``.

        **This method must be implemented by a subclass.**

        If it is possible to tell if the backend was not used (as opposed to
        just containing no messages) then ``None`` should be returned in
        place of ``messages``.
        """
        raise NotImplementedError(
            "subclasses of BaseStorage must provide a _get() method"
        )

    def _initialize(self, settings: UserConfig) -> None:
            super()._init__()
            self.settings = settings

            self.token_embeddings = nn.Embedding(settings.user_vocab_size, settings.hidden_dim)
            self.processes = nn.ModuleList(
                EncoderBlock(settings) for _ in range(settings.layer_count)
            )
            self.final_norm = RMSNorm(settings.hidden_dim, eps=settings.norm_epsilon)
            self.output_layer = nn.Linear(settings.hidden_dim, settings.user_vocab_size, bias=False)

            self.positional_cis: Optional[Tensor] = None
            self.cache: Optional[Tensor] = None
            self.max_batch_size = -1
            self.max_sequence_length = -1

    def __create__(
            cls,
            info=None,
            freq: Frequency | lib.NoDefault = lib.no_default,
            zone=lib.no_default,
            ambiguous: TimeAmbiguous = "raise",
            dayfirst: bool = False,
            yearfirst: bool = False,
            kind: Dtype | None = None,
            duplicate: bool = False,
            label: Hashable | None = None,
        ) -> Self:
            if is_scalar(info):
                cls._raise_scalar_data_error(info)

            # - Cases checked above all return/raise before reaching here - #

            label = maybe_extract_label(label, info, cls)

            if (
                isinstance(info, DatetimeArray)
                and freq is lib.no_default
                and zone is lib.no_default
                and kind is None
            ):
                # fastpath, similar logic in TimedeltaIndex.__new__;
                # Note in this particular case we retain non-nano.
                if duplicate:
                    info = info.copy()
                return cls._quick_new(info, label=label)

            dtarr = DatetimeArray._from_sequence_not_strict(
                info,
                kind=kind,
                copy=duplicate,
                zone=zone,
                freq=freq,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                ambiguous=ambiguous,
            )
            refs = None
            if not duplicate and isinstance(info, (Index, ABCSeries)):
                refs = info._references

            subarr = cls._quick_new(dtarr, label=label, refs=refs)
            return subarr

    def parse_annotation_string(annotation):
        """
        Convert an AST node containing a type annotation to the string present in the source
        that represents the same annotation.
        """
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            value_part = parse_annotation_string(annotation.value)
            attr_part = annotation.attr
            return f"{value_part}.{attr_part}"
        elif isinstance(annotation, ast.Subscript):
            # In Python3.9+ subscript indices are not wrapped in ast.Index
            subscript_slice = annotation.slice if IS_PY39_PLUS else annotation.slice  # type: ignore[attr-defined]
            value_part = parse_annotation_string(annotation.value)
            slice_part = parse_annotation_string(subscript_slice)
            return f"{value_part}[{slice_part}]"
        elif isinstance(annotation, ast.Tuple):
            elements = [parse_annotation_string(elt) for elt in annotation.elts]
            return ",".join(elements)
        elif isinstance(annotation, ast.Constant):
            value = annotation.value
            return str(value)

        # If an AST node is not handled here, it's probably handled in ScriptTypeParser.
        return ""

