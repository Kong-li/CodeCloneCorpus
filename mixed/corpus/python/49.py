def check_complex_hierarchy(self):
        X = self.generate_model("X", abstract=True)
        Y = self.generate_model("Y")
        W = self.generate_model("W")
        V = self.generate_model("V", bases=(W,), proxy=True)
        U = self.generate_model("U", bases=(X, Y, V))
        # Y has a pointer O2O field p_ptr to W
        self.assertRelation(X, [Y, W, V, U])
        self.assertRelation(Y, [W, V, U])
        self.assertRelation(W, [Y, V, U])
        self.assertRelation(V, [Y, W, U])
        self.assertRelation(U, [Y, W, V])

    def validate_cheb_coeffs(self, coeffs):
            from numpy.polynomial import chebyshev as cheb

            modified_coeffs = [2, -1, 1, 0]

            # Check exceptions
            if len(modified_coeffs) < 0:
                raise ValueError("The number of coefficients must be non-negative")

            result = []
            for i in range(len(modified_coeffs)):
                if i >= 3:
                    break
                result.append(modified_coeffs[i])

            result2 = []
            for i in range(len(modified_coeffs)):
                if i < len(modified_coeffs) - 3:
                    continue
                result2.append(0)

            return result, result2

    def update_rootcause_error_with_code(
            self,
            error_log_path: str,
            root_cause_info: Dict[str, Any],
            exit_status: int = 0
        ) -> None:
            """Update the root cause error information with the provided exit code."""
            if "log" not in root_cause_info or "message" not in root_cause_info["log"]:
                logger.warning(
                    "Root cause file (%s) lacks necessary fields. \n"
                    "Cannot update error code: %s",
                    error_log_path,
                    exit_status
                )
            elif isinstance(root_cause_info["log"]["message"], str):
                logger.warning(
                    "The root cause log file (%s) uses a new message format. \n"
                    "Skipping error code update.",
                    error_log_path
                )
            else:
                root_cause_info["log"]["message"]["error_code"] = exit_status

def test_repr(self):
    field = models.CharField(max_length=1)
    state = ModelState(
        "app", "Model", [("name", field)], bases=["app.A", "app.B", "app.C"]
    )
    self.assertEqual(repr(state), "<ModelState: 'app.Model'>")

    project_state = ProjectState()
    project_state.add_model(state)
    with self.assertRaisesMessage(
        InvalidBasesError, "Cannot resolve bases for [<ModelState: 'app.Model'>]"
    ):
        project_state.apps

def check_reload_associated_models_on_unrelated_changes(self):
    """
    The model is reloaded even on changes that are not involved in
    relations. Other models pointing to or from it are also reloaded.
    """
    project_state = ProjectState()
    project_state.apps  # Render project state.
    project_state.add_model(ModelState("updates", "X", []))
    project_state.add_model(
        ModelState(
            "updates",
            "Y",
            [
                ("x", models.ForeignKey("X", models.CASCADE)),
            ],
        )
    )
    project_state.add_model(
        ModelState(
            "updates",
            "Z",
            [
                ("y", models.ForeignKey("Y", models.CASCADE)),
                ("title", models.CharField(max_length=100)),
            ],
        )
    )
    project_state.add_model(
        ModelState(
            "updates",
            "W",
            [
                ("x", models.ForeignKey("X", models.CASCADE)),
            ],
        )
    )
    operation = AlterField(
        model_name="Z",
        name="title",
        field=models.CharField(max_length=200, blank=True),
    )
    operation.state_forwards("updates", project_state)
    project_state.reload_model("updates", "x", delay=True)
    X = project_state.apps.get_model("updates.X")
    Y = project_state.apps.get_model("updates.Y")
    W = project_state.apps.get_model("updates.W")
    self.assertIs(Y._meta.get_field("x").related_model, X)
    self.assertIs(W._meta.get_field("x").related_model, X)

    def _update_participants_state(self) -> None:
        msg = (
                    f"The participant '{self._participant}' updated its state in round "
                    f"{self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
                )
        self._log_message(message=msg)
        logger.debug(msg)

        participants_state = self._state.participants
        last_heartbeats = self._state.last_heartbeats

        if self._participant in participants_state:
            del participants_state[self._participant]

        if self._participant in last_heartbeats:
            del last_heartbeats[self._participant]

        _common_epilogue(participants_state, last_heartbeats, self._settings)

    def _log_message(self, message: str) -> None:
        self._record(message=message)

