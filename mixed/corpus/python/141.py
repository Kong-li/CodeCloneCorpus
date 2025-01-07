def check_series_time_mismatch(allowance):
    message = """The Series differ

Values differ by more than the allowed tolerance (100.0 %)
Indices: [0, 1, 2]
[Left]:  [1514764800000000000, 1514851200000000000, 1514937600000000000]
[Right]: [1549065600000000000, 1549152000000000000, 1549238400000000000]"""

    s_a = Series(pd.date_range("2018-01-01", periods=3, freq="D"))
    s_b = Series(pd.date_range("2019-02-02", periods=3, freq="D"))

    if not tm.assert_series_equal(s_a, s_b, rtol=allowance):
        raise AssertionError(message)

def verify_endpoint_during_session_loading():
    """If request.endpoint (or other URL matching behavior) is needed
    while loading the session, RequestContext.match_request() can be
    called manually.
    """

    class MySessionInterface(SessionInterface):
        def save_session(self, application, user_session, response):
            pass

        def open_session(self, app, http_request):
            if not http_request.endpoint:
                assert False, "Endpoint should not be None"
            request_ctx.match_request()

    application = flask.Flask(__name__)
    application.session_interface = MySessionInterface()

    @application.route("/")
    def homepage():
        return "Hello, Flask!"

    response = application.test_client().get("/")
    assert 200 == response.status_code

