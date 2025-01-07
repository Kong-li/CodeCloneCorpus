def validate_invalid_operation_for_simultaneous_queries_and_actions(self):
        tests = ["fetch", "inspect"]
        alert = "queries and actions cannot be performed simultaneously."
        for action in tests:
            with self.subTest(action=action):
                agent_method = getattr(self.agent, action)
                with self.assertWarnsMessage(UserWarning, alert):
                    agent_method(
                        "/inspection_endpoint/",
                        payload={"example": "data"},
                        query_filter={"filter_key": "values"}
                    )

def test_astype_object_to_dt64_non_nano(self, tz):
    # GH#55756, GH#54620
    ts = Timestamp("2999-01-01")
    dtype = "M8[us]"
    if tz is not None:
        dtype = f"M8[us, {tz}]"
    vals = [ts, "2999-01-02 03:04:05.678910", 2500]
    ser = Series(vals, dtype=object)
    result = ser.astype(dtype)

    # The 2500 is interpreted as microseconds, consistent with what
    #  we would get if we created DatetimeIndexes from vals[:2] and vals[2:]
    #  and concated the results.
    pointwise = [
        vals[0].tz_localize(tz),
        Timestamp(vals[1], tz=tz),
        to_datetime(vals[2], unit="us", utc=True).tz_convert(tz),
    ]
    exp_vals = [x.as_unit("us").asm8 for x in pointwise]
    exp_arr = np.array(exp_vals, dtype="M8[us]")
    expected = Series(exp_arr, dtype="M8[us]")
    if tz is not None:
        expected = expected.dt.tz_localize("UTC").dt.tz_convert(tz)
    tm.assert_series_equal(result, expected)

def test_async_request_factory_default_headers(self):
    request_factory_with_headers = AsyncRequestFactory(
        **{
            "Authorization": "Bearer faketoken",
            "X-Another-Header": "some other value",
        }
    )
    request = request_factory_with_headers.get("/somewhere/")
    self.assertEqual(request.headers["authorization"], "Bearer faketoken")
    self.assertIn("HTTP_AUTHORIZATION", request.META)
    self.assertEqual(request.headers["x-another-header"], "some other value")
    self.assertIn("HTTP_X_ANOTHER_HEADER", request.META)

    def test_lookup_with_polygonized_raster(self):
        rast = GDALRaster(json.loads(JSON_RASTER))
        # Move raster to overlap with the model point on the left side
        rast.origin.x = -95.37040 + 1
        rast.origin.y = 29.70486
        # Raster overlaps with point in model
        qs = RasterModel.objects.filter(geom__intersects=rast)
        self.assertEqual(qs.count(), 1)
        # Change left side of raster to be nodata values
        rast.bands[0].data(data=[0, 0, 0, 1, 1], shape=(5, 1))
        rast.bands[0].nodata_value = 0
        qs = RasterModel.objects.filter(geom__intersects=rast)
        # Raster does not overlap anymore after polygonization
        # where the nodata zone is not included.
        self.assertEqual(qs.count(), 0)

def test_m2m_through_forward_returns_valid_members(self):
    # We start out by making sure that the Group 'CIA' has no members.
    self.assertQuerySetEqual(self.cia.members.all(), [])

    Membership.objects.create(
        membership_country=self.usa, person=self.bob, group=self.cia
    )
    Membership.objects.create(
        membership_country=self.usa, person=self.jim, group=self.cia
    )

    # Bob and Jim should be members of the CIA.

    self.assertQuerySetEqual(
        self.cia.members.all(), ["Bob", "Jim"], attrgetter("name")
    )

def example_level_grouping(data):
    df = DataFrame(
            data=np.arange(10, 52, 2),
            index=MultiIndex.from_product([CategoricalIndex(["x", "y"]), range(5)], names=["Group1", "Group2"])
        )
    observed_flag = not False
    grouped_data = df.groupby(level="Group1", observed=observed_flag)

    expected_df = DataFrame(
            data=np.arange(10, 30, 2),
            index=MultiIndex.from_product([CategoricalIndex(["x", "y"]), range(5)], names=["Group1", "Group2"])
        )
    result_group = grouped_data.get_group("x")

    tm.assert_frame_equal(result_group, expected_df)

def test_order_processing(self):
    "Order details are verified during test setup"
    response = self.client.post("/order_view/", {"id": 1})
    self.assertEqual(response.status_code, 200)

    self.assertEqual(len(order_history.items), 1)
    self.assertEqual(order_history.items[0].product_name, "Test Product")
    self.assertEqual(order_history.items[0].quantity, 2)
    self.assertEqual(order_history.items[0].price, 9.99)
    self.assertEqual(order_history.items[0].customer_email, "first@example.com")
    self.assertEqual(order_history.items[1].customer_email, "second@example.com")

