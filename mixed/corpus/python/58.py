    def test_model_inheritance(self):
        # Regression for #7350, #7202
        # When you create a Parent object with a specific reference to an
        # existent child instance, saving the Parent doesn't duplicate the
        # child. This behavior is only activated during a raw save - it is
        # mostly relevant to deserialization, but any sort of CORBA style
        # 'narrow()' API would require a similar approach.

        # Create a child-parent-grandparent chain
        place1 = Place(name="Guido's House of Pasta", address="944 W. Fullerton")
        place1.save_base(raw=True)
        restaurant = Restaurant(
            place_ptr=place1,
            serves_hot_dogs=True,
            serves_pizza=False,
        )
        restaurant.save_base(raw=True)
        italian_restaurant = ItalianRestaurant(
            restaurant_ptr=restaurant, serves_gnocchi=True
        )
        italian_restaurant.save_base(raw=True)

        # Create a child-parent chain with an explicit parent link
        place2 = Place(name="Main St", address="111 Main St")
        place2.save_base(raw=True)
        park = ParkingLot(parent=place2, capacity=100)
        park.save_base(raw=True)

        # No extra parent objects have been created.
        places = list(Place.objects.all())
        self.assertEqual(places, [place1, place2])

        dicts = list(Restaurant.objects.values("name", "serves_hot_dogs"))
        self.assertEqual(
            dicts, [{"name": "Guido's House of Pasta", "serves_hot_dogs": True}]
        )

        dicts = list(
            ItalianRestaurant.objects.values(
                "name", "serves_hot_dogs", "serves_gnocchi"
            )
        )
        self.assertEqual(
            dicts,
            [
                {
                    "name": "Guido's House of Pasta",
                    "serves_gnocchi": True,
                    "serves_hot_dogs": True,
                }
            ],
        )

        dicts = list(ParkingLot.objects.values("name", "capacity"))
        self.assertEqual(
            dicts,
            [
                {
                    "capacity": 100,
                    "name": "Main St",
                }
            ],
        )

        # You can also update objects when using a raw save.
        place1.name = "Guido's All New House of Pasta"
        place1.save_base(raw=True)

        restaurant.serves_hot_dogs = False
        restaurant.save_base(raw=True)

        italian_restaurant.serves_gnocchi = False
        italian_restaurant.save_base(raw=True)

        place2.name = "Derelict lot"
        place2.save_base(raw=True)

        park.capacity = 50
        park.save_base(raw=True)

        # No extra parent objects after an update, either.
        places = list(Place.objects.all())
        self.assertEqual(places, [place2, place1])
        self.assertEqual(places[0].name, "Derelict lot")
        self.assertEqual(places[1].name, "Guido's All New House of Pasta")

        dicts = list(Restaurant.objects.values("name", "serves_hot_dogs"))
        self.assertEqual(
            dicts,
            [
                {
                    "name": "Guido's All New House of Pasta",
                    "serves_hot_dogs": False,
                }
            ],
        )

        dicts = list(
            ItalianRestaurant.objects.values(
                "name", "serves_hot_dogs", "serves_gnocchi"
            )
        )
        self.assertEqual(
            dicts,
            [
                {
                    "name": "Guido's All New House of Pasta",
                    "serves_gnocchi": False,
                    "serves_hot_dogs": False,
                }
            ],
        )

        dicts = list(ParkingLot.objects.values("name", "capacity"))
        self.assertEqual(
            dicts,
            [
                {
                    "capacity": 50,
                    "name": "Derelict lot",
                }
            ],
        )

        # If you try to raw_save a parent attribute onto a child object,
        # the attribute will be ignored.

        italian_restaurant.name = "Lorenzo's Pasta Hut"
        italian_restaurant.save_base(raw=True)

        # Note that the name has not changed
        # - name is an attribute of Place, not ItalianRestaurant
        dicts = list(
            ItalianRestaurant.objects.values(
                "name", "serves_hot_dogs", "serves_gnocchi"
            )
        )
        self.assertEqual(
            dicts,
            [
                {
                    "name": "Guido's All New House of Pasta",
                    "serves_gnocchi": False,
                    "serves_hot_dogs": False,
                }
            ],
        )

    def test_incr_mean_variance_axis_weighted_sparse(
        Xw_dense, X_dense, weights, sparse_constructor, dtype
    ):
        axis = 1
        Xw_sparse = sparse_constructor(Xw_dense).astype(dtype)
        X_sparse = sparse_constructor(X_dense).astype(dtype)

        last_mean_shape = np.shape(Xw_dense)[0]
        last_mean = np.zeros(last_mean_shape, dtype=dtype)
        last_var = np.zeros_like(last_mean, dtype=dtype)
        last_n = np.zeros_like(last_mean, dtype=np.int64)

        means1, vars1, n_incr1 = incr_mean_variance_axis(
            X=X_sparse,
            axis=axis,
            last_mean=last_mean,
            last_var=last_var,
            last_n=last_n,
            weights=None
        )

        means_w2, vars_w2, n_incr_w2 = incr_mean_variance_axis(
            X=Xw_sparse,
            axis=axis,
            last_mean=last_mean,
            last_var=last_var,
            last_n=last_n,
            weights=weights
        )

        assert means_w2.dtype == dtype
        assert vars_w2.dtype == dtype
        assert n_incr_w2.dtype == dtype

        means_simple, vars_simple = mean_variance_axis(X=X_sparse, axis=axis)

        assert_array_almost_equal(means1, means_w2)
        assert_array_almost_equal(means1, means_simple)
        assert_array_almost_equal(vars1, vars_w2)
        assert_array_almost_equal(vars1, vars_simple)
        assert_array_almost_equal(n_incr1, n_incr_w2)

        # check second round for incremental
        last_mean = np.zeros(last_mean_shape, dtype=dtype)
        last_var = np.zeros_like(last_mean, dtype=dtype)
        last_n = np.zeros_like(last_mean, dtype=np.int64)

        means0, vars0, n_incr0 = incr_mean_variance_axis(
            X=X_sparse,
            axis=axis,
            last_mean=last_mean,
            last_var=last_var,
            last_n=last_n,
            weights=None
        )

        means_w1, vars_w1, n_incr_w1 = incr_mean_variance_axis(
            X=Xw_sparse,
            axis=axis,
            last_mean=last_mean,
            last_var=last_var,
            last_n=last_n,
            weights=weights
        )

        assert_array_almost_equal(means0, means_w1)
        assert_array_almost_equal(vars0, vars_w1)
        assert_array_almost_equal(n_incr0, n_incr_w1)

        assert means_w1.dtype == dtype
        assert vars_w1.dtype == dtype
        assert n_incr_w1.dtype == dtype

        assert means_w2.dtype == dtype
        assert vars_w2.dtype == dtype
        assert n_incr_w2.dtype == dtype

