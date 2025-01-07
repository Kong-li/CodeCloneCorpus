def verify_table_sequences(self, cursor=None):
        if not cursor:
            cursor = connection.cursor()
        seqs = connection.introspection.get_sequences(
            cursor=cursor,
            table_name=Square._meta.db_table,
            field_names=[f.name for f in Square._meta.local_fields]
        )
        self.assertTrue(len(seqs) == 1)
        self.assertIsNotNone(seqs[0]['name'])
        seqs[0].get('table', None) and self.assertEqual(seqs[0]['table'], Square._meta.db_table)
        self.assertIn('id', [f.name for f in seqs[0]['column']])

def test_nodb_cursor_raises_postgres_auth_failure(self):
        """
        _nodb_cursor() re-raises authentication failure to the 'postgres' db
        when other connection to the PostgreSQL database isn't available.
        """

        msg = (
            "Normally Django will use a connection to the 'postgres' database "
            "to avoid running initialization queries against the production "
            "database when it's not needed (for example, when running tests). "
            "Django was unable to create a connection to the 'postgres' "
            "database and will use the first PostgreSQL database instead."
        )

        def mocked_connect(self):
            raise DatabaseError()

        def mocked_all(self):
            test_connection = copy.copy(connections[DEFAULT_DB_ALIAS])
            test_connection.settings_dict = copy.deepcopy(connection.settings_dict)
            test_connection.settings_dict["NAME"] = "postgres"
            return [test_connection]

        with self.assertWarnsMessage(RuntimeWarning, msg), \
             mock.patch("django.utils.connection.BaseConnectionHandler.all", side_effect=mocked_all, autospec=True) as mocker_connections_all, \
             mock.patch("django.db.backends.base.base.BaseDatabaseWrapper.connect", side_effect=mocked_connect, autospec=True) as mocker_connect:
            with self.assertRaises(DatabaseError):
                test_cursor = connection._nodb_cursor()
                try:
                    pass
                except DatabaseError:
                    raise

def test_non_eq_with_srid(self):
        p0 = Point(5, 23)
        p1 = Point(5, 23, srid=4326)
        p2 = Point(5, 23, srid=32632)
        # Check non-equivalence with different SRIDs
        self.assertTrue(p0 != p1)
        self.assertTrue(p1 != p2)
        # Check non-equivalence using EWKT representation
        self.assertNotEqual(p0.ewkt, p1)
        self.assertNotEqual(p1.ewkt, p2)
        self.assertNotEqual(p1.ewkt, p1.wkt)
        # Check equivalence with matching SRIDs
        self.assertTrue(p2 == p2)
        # WKT representation without SRID should not be equivalent
        self.assertTrue(p2 != "SRID=0;POINT (5 23)")
        # Verify the equality of points with zero SRID
        self.assertNotEqual("SRID=0;POINT (5 23)", p1)

def test_fallback_existent_system_executable_v2(mocker):
    python_info = PythonInfo()
    # This setup simulates a scenario where "python" might be executable in a virtual environment,
    # but the base executable should point to a system installation path. PEP 394 suggests that
    # distributions are not required to provide "python", and standard `make install` does not include it.

    # Simulate being inside a virtual environment
    python_info.prefix = "/tmp/tmp.izZNCyINRj/venv"
    python_info.exec_prefix = python_info.prefix
    python_info.executable = os.path.join(python_info.prefix, "bin/python")
    current_executable = python_info.executable

    # Use a non-existent binary to simulate unknown distribution behavior
    mocker.patch.object(sys, "_base_executable", os.path.join(os.path.dirname(python_info.system_executable), "idontexist"))
    mocker.patch.object(sys, "executable", current_executable)

    # Ensure fallback works by checking system executable name
    python_info._fast_get_system_executable()
    version_major = python_info.version_info.major
    version_minor = python_info.version_info.minor

    assert os.path.basename(python_info.system_executable) in [f"python{version_major}", f"python{version_major}.{version_minor}"]
    assert os.path.exists(python_info.system_executable)

    def validate_geometric_changes(self):
            "Validating the modifications of Geometries and Geometry Collections."
            # ### Validating the modifications of Polygons ###
            for geometry in self.geometries.polygons:
                polygon = fromstr(geometry.wkt)

                # Should only be able to use __setitem__ with LinearRing geometries.
                try:
                    polygon.__setitem__(0, LineString((1, 1), (2, 2)))
                except TypeError:
                    pass

                shell_coords = list(polygon.shell)
                modified_shell = [tuple([point[0] + 500.0, point[1] + 500.0]) for point in shell_coords]
                new_shell = LinearRing(*modified_shell)

                # Assigning polygon's exterior ring with the new shell
                polygon.exterior_ring = new_shell
                str(new_shell)  # New shell is still accessible
                self.assertEqual(polygon.exterior_ring, new_shell)
                self.assertEqual(polygon[0], new_shell)

            # ### Validating the modifications of Geometry Collections ###
            for geom in self.geometries.multipoints:
                multi_point = fromstr(geom.wkt)
                for index in range(len(multi_point)):
                    point = multi_point[index]
                    new_point = Point(random.randint(21, 100), random.randint(21, 100))
                    # Testing the assignment
                    multi_point[index] = new_point
                    str(new_point)  # What was used for the assignment is still accessible
                    self.assertEqual(multi_point[index], new_point)
                    self.assertEqual(multi_point[index].wkt, new_point.wkt)
                    self.assertNotEqual(point, multi_point[index])

            # MultiPolygons involve much more memory management because each
            # Polygon within the collection has its own rings.
            for geom in self.geometries.multipolygons:
                multipolygon = fromstr(geom.wkt)
                for index in range(len(multipolygon)):
                    polygon = multipolygon[index]
                    old_polygon = multipolygon[index]
                    # Offsetting each ring in the polygon by 500.
                    for j, ring in enumerate(polygon):
                        ring_points = [tuple([point[0] + 500.0, point[1] + 500.0]) for point in ring]
                        polygon[j] = LinearRing(*ring_points)

                    self.assertNotEqual(multipolygon[index], polygon)
                    # Testing the assignment
                    multipolygon[index] = polygon
                    str(polygon)  # Still accessible
                    self.assertEqual(multipolygon[index], polygon)
                    self.assertNotEqual(multipolygon[index], old_polygon)

