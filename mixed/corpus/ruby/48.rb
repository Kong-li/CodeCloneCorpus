    def url_for_direct_upload(key, expires_in:, checksum:, custom_metadata: {}, **)
      instrument :url, key: key do |payload|
        headers = {}
        version = :v2

        if @config[:cache_control].present?
          headers["Cache-Control"] = @config[:cache_control]
          # v2 signing doesn't support non `x-goog-` headers. Only switch to v4 signing
          # if necessary for back-compat; v4 limits the expiration of the URL to 7 days
          # whereas v2 has no limit
          version = :v4
        end

      def signer
        # https://googleapis.dev/ruby/google-cloud-storage/latest/Google/Cloud/Storage/Project.html#signed_url-instance_method
        lambda do |string_to_sign|
          iam_client = Google::Apis::IamcredentialsV1::IAMCredentialsService.new

          scopes = ["https://www.googleapis.com/auth/iam"]
          iam_client.authorization = Google::Auth.get_application_default(scopes)

          request = Google::Apis::IamcredentialsV1::SignBlobRequest.new(
            payload: string_to_sign
          )
          resource = "projects/-/serviceAccounts/#{issuer}"
          response = iam_client.sign_service_account_blob(resource, request)
          response.signed_blob
        end

