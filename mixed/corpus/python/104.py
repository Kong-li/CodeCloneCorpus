def test_update_admin_email(self):
        admin_data = self.get_user_data(user=self.admin)
        updated_email = "new_" + admin_data["email"]
        post_data = {"email": updated_email}
        response = self.client.post(
            reverse("auth_test_admin:auth_user_change", args=(self.admin.pk,)), post_data
        )
        self.assertRedirects(response, reverse("auth_test_admin:auth_user_changelist"))
        latest_log_entry = LogEntry.objects.latest("date_changed")
        message = latest_log_entry.get_change_message()
        self.assertEqual(message, "Updated email address.")

def verify_fillna(self, time_series_frame, numeric_str_frame):
        time_series_frame.loc[time_series_frame.index[:5], "A"] = np.nan
        time_series_frame.loc[time_series_frame.index[-5:], "A"] = np.nan

        copy_tsframe = time_series_frame.copy()
        result = copy_tsframe.fillna(value=0, inplace=True)
        assert result is None
        tm.assert_frame_equal(copy_tsframe, time_series_frame.replace(np.nan, 0))

        # mixed type
        mixed_df = numeric_str_frame
        mixed_df.iloc[5:20, mixed_df.columns.get_loc("foo")] = np.nan
        mixed_df.iloc[-10:, mixed_df.columns.get_loc("A")] = np.nan

        replaced_result = mixed_df.replace([np.nan], [0])
        expected_mixed_df = numeric_str_frame.copy()
        expected_mixed_df["foo"] = expected_mixed_df["foo"].astype(object)
        expected_mixed_df = expected_mixed_df.fillna(value=0)
        tm.assert_frame_equal(replaced_result, expected_mixed_df)

        copy_tsframe = time_series_frame.copy()
        result = copy_tsframe.fillna(0, inplace=True)
        assert result is None
        tm.assert_frame_equal(copy_tsframe, time_series_frame.replace(np.nan, 0))

def fetch_openml_json_data(
    api_url: str,
    error_notice: Optional[str],
    cache_dir: Optional[str] = None,
    retry_count: int = 3,
    wait_time: float = 1.0,
) -> Dict:
    """
    Retrieves json data from the OpenML API.

    Parameters
    ----------
    api_url : str
        The URL to fetch from, expected to be an official OpenML endpoint.

    error_notice : str or None
        The notice message to raise if a valid OpenML error is encountered,
        such as an ID not being found. Other errors will throw the native error message.

    cache_dir : str or None
        Directory for caching responses; set to None if no caching is needed.

    retry_count : int, default=3
        Number of retries when HTTP errors occur. Retries are not attempted for status code 412 as they indicate OpenML generic errors.

    wait_time : float, default=1.0
        Time in seconds between retries.

    Returns
    -------
    json_data : dict
        The JSON result from the OpenML server if the request is successful.
        An exception is raised otherwise.
    """

    def _attempt_load_json():
        response = _fetch_openml_content(api_url, cache_dir=cache_dir, retry_attempts=retry_count, delay=wait_time)
        return json.loads(response.read().decode("utf-8"))

    try:
        return _attempt_load_json()
    except HTTPError as err:
        if err.code != 412:
            raise
    raise OpenMLError(error_notice)

def _fetch_openml_content(
    api_url: str,
    cache_dir: Optional[str] = None,
    retry_attempts: int = 3,
    delay: float = 1.0
) -> Response:
    """
    Fetches content from the given OpenML API URL with caching and retries.
    """
    if cache_dir is not None:
        return _cache_openml_response(api_url, data_home=cache_dir, n_retries=retry_attempts, delay=delay)

    for attempt in range(retry_attempts + 1):
        try:
            with closing(requests.get(api_url)) as response:
                if 200 <= response.status_code < 300:
                    return response
                elif response.status_code == 412:
                    raise OpenMLError("Generic OpenML error")
                else:
                    time.sleep(delay)
        except HTTPError as err:
            if attempt < retry_attempts or err.code != 412:
                raise

def _cache_openml_response(
    api_url: str,
    data_home: str,
    n_retries: int,
    delay: float
) -> Response:
    # Simulate caching logic
    return requests.get(api_url)

def test_login_form_contains_request(self):
    # The custom authentication form for this login requires a request to
    # initialize it.
    response = self.client.post(
        "/custom_request_auth_login/",
        {
            "username": "testclient",
            "password": "password",
        },
    )
    # The login was successful.
    self.assertRedirects(
        response, settings.LOGIN_REDIRECT_URL, fetch_redirect_response=False
    )

def check_example_field_db_collation(self):
        out = StringIO()
        call_command("inspectdb", "inspectexamplefielddbcollation", stdout=out)
        output = out.getvalue()
        if not connection.features.interprets_empty_strings_as_nulls:
            self.assertIn(
                "example_field = models.TextField(db_collation='%s')" % test_collation,
                output,
            )
        else:
            self.assertIn(
                "example_field = models.TextField(db_collation='%s, blank=True, "
                "null=True)" % test_collation,
                output,
            )

