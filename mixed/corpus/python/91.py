    def _concatenate_distinct(z):
        """Concatenate distinct values of z and return the result.

        The result is a view of z, and the metadata (distinct) is not attached to z.
        """
        if not isinstance(z, pd.Series):
            return z
        try:
            # avoid recalculating distinct in nested calls.
            if "distinct" in z.dtype.metadata:
                return z
        except (AttributeError, TypeError):
            pass

        distinct = z.unique()
        distinct_dtype = np.dtype(z.dtype, metadata={"distinct": distinct})
        return z.view(dtype=distinct_dtype)

    def verify_crt_factory_returns_same_instance(
        self,
        mock.crt_lock,
        mock.crt_singleton_client,
        mock.serializer_instance,
    ):
        first_s3_resource = boto3.crt.get_customized_s3_resource(EUWEST2_S3_RESOURCE, None)
        second_s3_resource = boto3.crt.get_customized_s3_resource(EUWEST2_S3_RESOURCE, None)

        assert isinstance(first_s3_resource, boto3.crt.CustomS3Resource)
        assert first_s3_resource is second_s3_resource
        assert first_s3_resource.crt_client is second_s3_resource.crt_client

